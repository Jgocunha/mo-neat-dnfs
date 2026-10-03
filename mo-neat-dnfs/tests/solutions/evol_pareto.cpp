#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "neat/pareto.h"
#include "neat/population.h"
#include "neat_tools/ablation_presets.h"
#include "neat_tools/resource_paths.h"
#include "neat_tools/solution_registry.h"
#include "test_evolution_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

namespace
{
    // A task's own config with the pareto-selection preset over it, as
    // `mo-neat-dnfs-evol --task <slug> --ablation pareto-selection` runs it.
    // ScopedTaskConfig resets SelectionConstants; the label is reset here.
    class ScopedParetoPreset
    {
    public:
        explicit ScopedParetoPreset(const std::string& slug) : taskConfig(slug)
        {
            REQUIRE(AblationPresets::applyByName("pareto-selection"));
        }
        ~ScopedParetoPreset() { AblationConstants::reset(); }
        ScopedParetoPreset(const ScopedParetoPreset&) = delete;
        ScopedParetoPreset& operator=(const ScopedParetoPreset&) = delete;
        ScopedParetoPreset(ScopedParetoPreset&&) = delete;
        ScopedParetoPreset& operator=(ScopedParetoPreset&&) = delete;

    private:
        ScopedTaskConfig taskConfig;
    };

    std::unordered_set<std::string> runDirectoryNames(const std::filesystem::path& parentDirectory)
    {
        std::unordered_set<std::string> names;
        if (std::filesystem::exists(parentDirectory))
        {
            for (const auto& entry : std::filesystem::directory_iterator(parentDirectory))
            {
                names.insert(entry.path().filename().string());
            }
        }
        return names;
    }

    std::filesystem::path newRunDirectory(const std::filesystem::path& parentDirectory,
        const std::unordered_set<std::string>& preExisting)
    {
        for (const auto& entry : std::filesystem::directory_iterator(parentDirectory))
        {
            if (!preExisting.contains(entry.path().filename().string()))
            {
                return entry.path();
            }
        }
        return {};
    }

    std::vector<nlohmann::json> readJsonLines(const std::filesystem::path& path)
    {
        std::ifstream file(path);
        std::vector<nlohmann::json> records;
        for (std::string line; std::getline(file, line);)
        {
            if (!line.empty())
            {
                records.push_back(nlohmann::json::parse(line));
            }
        }
        return records;
    }

    RankedPoint rankedPointOf(const nlohmann::json& individual)
    {
        return { individual.at("objectives").get<std::vector<double>>(), individual.at("violation").get<double>() };
    }

    // The recorded ranks are exactly nonDominatedSort() of the recorded points. Checking
    // "a dominates b => a ranks lower" instead would be wrong: with epsilon and the violation
    // tie band, dominance can be cyclic, and the sort's documented fallback then puts
    // points that dominate each other on one front.
    void requireRanksMatchTheSort(const nlohmann::json& record, const double epsilon, const double violationEpsilon)
    {
        const auto& individuals = record.at("individuals");
        std::vector<RankedPoint> points;
        for (const auto& individual : individuals)
        {
            points.push_back(rankedPointOf(individual));
        }
        const auto fronts = nonDominatedSort(points, epsilon, violationEpsilon);
        for (size_t rank = 0; rank < fronts.size(); ++rank)
        {
            for (const size_t index : fronts[rank])
            {
                REQUIRE(individuals[index].at("rank").get<size_t>() == rank);
            }
        }
    }

    // The archive keeps feasible points only and never empties once it holds one,
    // so it is non-empty exactly from the first generation that had a feasible
    // individual. Whether a short run reaches feasibility is up to the search, so
    // the test asserts this equivalence rather than a non-empty archive.
    void requireArchiveFollowsFeasibility(const std::vector<nlohmann::json>& records)
    {
        bool feasibleSeen = false;
        for (const auto& record : records)
        {
            const auto& individuals = record.at("individuals");
            feasibleSeen = feasibleSeen || std::ranges::any_of(individuals,
                [](const nlohmann::json& individual) { return individual.at("violation").get<double>() == 0.0; });
            INFO("generation " << record.at("generation"));
            REQUIRE((record.at("archive").at("size").get<size_t>() > 0) == feasibleSeen);
        }
    }

    // Runs settings.numRuns Pareto-mode evolutions of the task with file output on,
    // checks each run's objectives.jsonl, and deletes the run's directory.
    void requireParetoEvolutionHolds(const std::string& slug, const std::string& solutionName)
    {
        const ScopedParetoPreset preset(slug);
        const auto* task = findTask(slug);
        REQUIRE(task != nullptr);
        const EvolutionRunSettings settings;
        const auto parentDirectory = paths::dataRoot() / "data" / (solutionName + AblationConstants::label);

        for (int run = 0; run < settings.numRuns; ++run)
        {
            INFO("run " << run);
            resetGlobalState();
            const auto preExisting = runDirectoryNames(parentDirectory);
            const PopulationParameters parameters(settings.populationSize, settings.numGenerations,
                settings.targetFitness, settings.parallel);
            Population population{ parameters, task->makeFromTopology(defaultTopologyFor(*task)) };
            population.initialize();
            REQUIRE_NOTHROW(population.evolve());

            // Elitism is unchanged in Pareto mode (decision D4). validateElitism() runs under
            // the test binary's Throw policy, so a clean report covers it. The recorded best
            // fitness itself is not monotone: the preserved elite is re-evaluated every
            // generation, and noise can move it by more than elitismFitnessEpsilon.
            INFO("validation:\n" << join(population.getValidationReport().messages, "\n"));
            REQUIRE(population.getValidationReport().clean());

            const auto runDirectory = newRunDirectory(parentDirectory, preExisting);
            REQUIRE_FALSE(runDirectory.empty());
            const auto records = readJsonLines(runDirectory / "objectives.jsonl");
            REQUIRE(static_cast<int>(records.size()) == population.getCurrentGeneration());
            for (const auto& record : records)
            {
                INFO("generation " << record.at("generation"));
                REQUIRE(record.at("mode") == "pareto");
                REQUIRE(record.at("objectiveGroups").get<std::vector<std::vector<size_t>>>() == SelectionConstants::objectiveGroups);
                REQUIRE(static_cast<int>(record.at("individuals").size()) == settings.populationSize);
                requireRanksMatchTheSort(record, SelectionConstants::dominanceEpsilon,
                    SelectionConstants::violationEpsilon);
            }
            requireArchiveFollowsFeasibility(records);
            std::filesystem::remove_all(runDirectory);
        }
        // An empty data/<Task> Pareto/ would show up in the dashboard as an experiment with no runs.
        if (std::filesystem::is_empty(parentDirectory))
        {
            std::filesystem::remove(parentDirectory);
        }
    }
}

TEST_CASE("XOR evolves under the pareto-selection preset with ranks that match the sort",
    "[Evolution][Solutions][XOR][Pareto]")
{
    requireParetoEvolutionHolds("xor", "XOR");
}

TEST_CASE("AND evolves under the pareto-selection preset with ranks that match the sort",
    "[Evolution][Solutions][AND][Pareto]")
{
    requireParetoEvolutionHolds("and", "AND");
}
