#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <unordered_set>

#include "neat/pareto.h"
#include "neat/population.h"
#include "neat/species.h"
#include "neat_tools/resource_paths.h"
#include "test_helpers.h"
#include "test_population_access.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

namespace
{
    constexpr double infinity = std::numeric_limits<double>::infinity();

    // Switches selection to Pareto mode with the given floor and epsilon, and
    // restores every SelectionConstants default afterwards even if the test fails.
    struct ScopedParetoSelection
    {
        explicit ScopedParetoSelection(const double floor = 0.0, const double epsilon = 0.0)
        {
            SelectionConstants::mode = SelectionMode::Pareto;
            SelectionConstants::feasibilityFloor = floor;
            SelectionConstants::dominanceEpsilon = epsilon;
        }
        ~ScopedParetoSelection() { SelectionConstants::reset(); }
        ScopedParetoSelection(const ScopedParetoSelection&) = delete;
        ScopedParetoSelection& operator=(const ScopedParetoSelection&) = delete;
        ScopedParetoSelection(ScopedParetoSelection&&) = delete;
        ScopedParetoSelection& operator=(ScopedParetoSelection&&) = delete;
    };

    // An initialized, evaluated solution at a chosen point of objective space.
    std::shared_ptr<FixedObjectivesSolution> evaluatedAt(const std::vector<double>& partials, const double fitness)
    {
        auto solution = std::make_shared<FixedObjectivesSolution>(makeTopology(1, 1), partials, fitness);
        solution->initialize();
        solution->evaluate();
        return solution;
    }

    // An evaluated solution placed directly on a front, bypassing the population's sort.
    std::shared_ptr<FixedObjectivesSolution> rankedAt(const std::vector<double>& partials, const double fitness,
        const int rank, const double crowding)
    {
        auto solution = evaluatedAt(partials, fitness);
        solution->setParetoRanking(rank, crowding, 0.0, 1.0 / (1.0 + rank));
        return solution;
    }

    // A population whose solutions sit at the given points of objective space,
    // evaluated but not yet ranked or speciated. File I/O is off.
    std::unique_ptr<Population> populationAt(const std::vector<std::vector<double>>& points)
    {
        resetGlobalState();
        const PopulationParameters parameters(static_cast<int>(points.size()), 10, 1.1, false);
        auto population = std::make_unique<Population>(parameters,
            std::make_shared<FixedObjectivesSolution>(makeTopology(1, 1), points.front(), 0.0), false);
        auto& solutions = PopulationTestAccess::solutions(*population);
        for (size_t i = 0; i < points.size(); ++i)
        {
            solutions[i] = std::make_shared<FixedObjectivesSolution>(makeTopology(1, 1), points[i], 0.0);
        }
        population->initialize();
        PopulationTestAccess::evaluate(*population);
        return population;
    }

    // Replaces solution @p index of @p population with a fresh one at @p point and re-evaluates.
    void moveSolutionTo(Population& population, const size_t index, const std::vector<double>& point)
    {
        auto& solutions = PopulationTestAccess::solutions(population);
        solutions[index] = std::make_shared<FixedObjectivesSolution>(makeTopology(1, 1), point, 0.0);
        solutions[index]->initialize();
        PopulationTestAccess::evaluate(population);
    }
}

// --- comparator (section 2, rows 1, 6, 7 and 9 all go through it) ---------------

TEST_CASE("Solution::isPreferredTo reduces to fitness > in scalar mode", "[ParetoSelection]")
{
    SelectionConstants::reset();
    std::mt19937 generator(20260930);
    std::uniform_int_distribution<int> tenths(0, 10);
    std::uniform_real_distribution<double> anyFitness(0.0, 1.0);

    std::vector<SolutionPtr> solutions;
    for (int i = 0; i < 24; ++i)
    {
        // Half on a coarse grid, so that many pairs tie exactly.
        const double fitness = i % 2 == 0 ? tenths(generator) / 10.0 : anyFitness(generator);
        const auto solution = evaluatedAt({ fitness }, fitness);
        // A Pareto ranking that disagrees with fitness must not matter in scalar mode.
        solution->setParetoRanking(i % 3, static_cast<double>(i), 0.0, 1.0 - fitness);
        solutions.push_back(solution);
    }

    for (const auto& a : solutions)
    {
        for (const auto& b : solutions)
        {
            REQUIRE(a->isPreferredTo(*b) == (a->getFitness() > b->getFitness()));
            REQUIRE(a->isEquivalentForSelection(*b) == (std::abs(a->getFitness() - b->getFitness()) < 1e-6));
        }
    }
}

TEST_CASE("Solution::isPreferredTo ranks by front, then crowding, in Pareto mode", "[ParetoSelection]")
{
    const ScopedParetoSelection pareto;
    const auto frontZero = rankedAt({ 0.2, 0.8 }, 0.1, 0, 0.5);
    const auto frontOne = rankedAt({ 0.9, 0.9 }, 0.9, 1, infinity);
    const auto frontZeroSparse = rankedAt({ 0.8, 0.2 }, 0.1, 0, 0.8);
    const auto unranked = evaluatedAt({ 1.0, 1.0 }, 1.0);

    REQUIRE(frontZero->isPreferredTo(*frontOne));
    REQUIRE_FALSE(frontOne->isPreferredTo(*frontZero));
    REQUIRE(frontZeroSparse->isPreferredTo(*frontZero));
    REQUIRE_FALSE(frontZero->isPreferredTo(*frontZeroSparse));
    REQUIRE_FALSE(frontZero->isPreferredTo(*frontZero));

    // A solution the population has not ranked yet (an offspring) loses to every ranked one.
    REQUIRE(frontOne->isPreferredTo(*unranked));
    REQUIRE_FALSE(unranked->isPreferredTo(*frontOne));
    REQUIRE_FALSE(unranked->isPreferredTo(*unranked));
}

TEST_CASE("Solution::isEquivalentForSelection in Pareto mode means same front and mutually non-dominated", "[ParetoSelection]")
{
    const ScopedParetoSelection pareto;
    const auto left = rankedAt({ 0.8, 0.2 }, 0.5, 0, infinity);
    const auto right = rankedAt({ 0.2, 0.8 }, 0.5, 0, 0.3);
    const auto dominatingSameFront = rankedAt({ 0.9, 0.9 }, 0.9, 0, 0.3);
    const auto nextFront = rankedAt({ 0.1, 0.9 }, 0.5, 1, infinity);

    REQUIRE(left->isEquivalentForSelection(*right));
    REQUIRE(right->isEquivalentForSelection(*left));
    // The same front index can hold a dominating pair after the cyclic-epsilon fallback.
    REQUIRE_FALSE(dominatingSameFront->isEquivalentForSelection(*left));
    REQUIRE_FALSE(left->isEquivalentForSelection(*dominatingSameFront));
    REQUIRE_FALSE(left->isEquivalentForSelection(*nextFront));
}

TEST_CASE("Solution::getSelectionFitness is the fitness in scalar mode and rank-derived in Pareto mode", "[ParetoSelection]")
{
    SelectionConstants::reset();
    const auto solution = evaluatedAt({ 0.3, 0.7 }, 0.42);
    solution->setParetoRanking(2, 0.5, 0.0, 0.25);

    REQUIRE(solution->getSelectionFitness() == Catch::Approx(0.42));

    const ScopedParetoSelection pareto;
    REQUIRE(solution->getSelectionFitness() == Catch::Approx(0.25));
}

// --- crossover (section 2, row 7) ------------------------------------------------

TEST_CASE("Solution::crossover takes the preferred parent as the fitter one in both modes", "[ParetoSelection]")
{
    resetGlobalState();
    const auto lowFitnessFrontZero = rankedAt({ 0.8, 0.2 }, 0.1, 0, infinity);
    const auto highFitnessFrontOne = rankedAt({ 0.7, 0.1 }, 0.9, 1, infinity);
    lowFitnessFrontZero->addFieldGene(FieldGene({ FieldGeneType::HIDDEN, 900 }));
    highFitnessFrontOne->addFieldGene(FieldGene({ FieldGeneType::HIDDEN, 901 }));

    SECTION("scalar mode: the higher fitness")
    {
        SelectionConstants::reset();
        const auto offspring = lowFitnessFrontZero->crossover(highFitnessFrontOne);
        REQUIRE(std::get<0>(offspring->getParents()) == highFitnessFrontOne->getId());
        for (const auto& gene : offspring->getGenome().getFieldGenes())
        {
            REQUIRE(highFitnessFrontOne->getGenome().containsFieldGene(gene));
        }
    }
    SECTION("Pareto mode: the lower front, whatever the fitness")
    {
        const ScopedParetoSelection pareto;
        const auto offspring = highFitnessFrontOne->crossover(lowFitnessFrontZero);
        REQUIRE(std::get<0>(offspring->getParents()) == lowFitnessFrontZero->getId());
        for (const auto& gene : offspring->getGenome().getFieldGenes())
        {
            REQUIRE(lowFitnessFrontZero->getGenome().containsFieldGene(gene));
        }
    }
}

TEST_CASE("Solution::crossover inherits disjoint genes at random only from equivalent parents", "[ParetoSelection]")
{
    resetGlobalState();
    const ConnectionGene disjointGene(ConnectionTuple(1, 2), 1);
    constexpr int crossovers = 64;

    // Counts the offspring of @p fitter x @p other that inherit @p fitter's only
    // (disjoint) connection gene. Equivalent parents take the random branch,
    // so some offspring lose it (the chance that all 64 keep it is 2^-64).
    const auto inheritedCount = [&](const SolutionPtr& fitter, const SolutionPtr& other)
    {
        int inherited = 0;
        for (int i = 0; i < crossovers; ++i)
        {
            if (fitter->crossover(other)->containsConnectionGene(disjointGene))
            {
                ++inherited;
            }
        }
        return inherited;
    };

    SECTION("scalar mode: equal fitness is equivalent, a different fitness is not")
    {
        SelectionConstants::reset();
        const auto fitter = evaluatedAt({ 0.5 }, 0.5);
        fitter->addConnectionGene(disjointGene);
        REQUIRE(inheritedCount(fitter, evaluatedAt({ 0.5 }, 0.5)) < crossovers);
        REQUIRE(inheritedCount(fitter, evaluatedAt({ 0.2 }, 0.2)) == crossovers);
    }
    SECTION("Pareto mode: a non-dominated partner on the same front is equivalent, the next front is not")
    {
        const ScopedParetoSelection pareto;
        const auto fitter = rankedAt({ 0.8, 0.2 }, 0.5, 0, infinity);
        fitter->addConnectionGene(disjointGene);
        REQUIRE(inheritedCount(fitter, rankedAt({ 0.2, 0.8 }, 0.1, 0, 0.3)) < crossovers);
        REQUIRE(inheritedCount(fitter, rankedAt({ 0.7, 0.1 }, 0.9, 1, infinity)) == crossovers);
    }
}

// --- species (section 2, rows 1 and 2) --------------------------------------------

TEST_CASE("Species sorts, prunes and picks its champion by the comparator in Pareto mode", "[ParetoSelection]")
{
    const auto bestFitnessWorstFront = rankedAt({ 0.1, 0.1 }, 0.9, 2, infinity);
    const auto frontZero = rankedAt({ 0.9, 0.1 }, 0.5, 0, infinity);
    const auto frontOne = rankedAt({ 0.5, 0.05 }, 0.1, 1, infinity);

    const auto speciesOf = [&]
    {
        auto species = std::make_shared<Species>();
        species->addSolution(bestFitnessWorstFront);
        species->addSolution(frontZero);
        species->addSolution(frontOne);
        return species;
    };

    SECTION("scalar mode keeps the fitness order")
    {
        SelectionConstants::reset();
        const auto species = speciesOf();
        species->assignChampion();
        REQUIRE(species->getChampion() == bestFitnessWorstFront);
        species->pruneWorsePerformingMembers(1.0 / 3.0);
        REQUIRE_FALSE(species->contains(frontOne));
    }
    SECTION("Pareto mode orders by front")
    {
        const ScopedParetoSelection pareto;
        const auto species = speciesOf();
        species->sortMembersByFitness();
        REQUIRE(species->getMembers() == std::vector<SolutionPtr>{ frontZero, frontOne, bestFitnessWorstFront });

        species->assignChampion(true);
        REQUIRE(species->getChampion() == frontZero);

        species->pruneWorsePerformingMembers(1.0 / 3.0);
        REQUIRE_FALSE(species->contains(bestFitnessWorstFront));
        REQUIRE(species->size() == 2);
    }
}

TEST_CASE("Species::assignChampion(bool) records the improvement it is given, not the champion's fitness", "[ParetoSelection]")
{
    const ScopedParetoSelection pareto;
    Species species;
    species.addSolution(rankedAt({ 0.5, 0.5 }, 0.5, 0, infinity));

    species.assignChampion(true);
    REQUIRE(species.hasFitnessImprovedOverTheLastGenerations());

    // A fitter member arrives, but the archive accepted nothing: still no improvement.
    species.addSolution(rankedAt({ 0.9, 0.9 }, 0.99, 0, infinity));
    for (int i = 0; i < PopulationConstants::generationsWithoutImprovementThresholdInSpecies; ++i)
    {
        species.assignChampion(false);
    }
    REQUIRE_FALSE(species.hasFitnessImprovedOverTheLastGenerations());

    species.assignChampion(true);
    REQUIRE(species.hasFitnessImprovedOverTheLastGenerations());
}

// --- population ------------------------------------------------------------------

TEST_CASE("Population ranks objectives in Pareto mode even without file output", "[ParetoSelection]")
{
    const auto population = populationAt({ { 0.5, 0.5 }, { 0.1, 0.1 } });
    SelectionConstants::reset();
    REQUIRE_FALSE(PopulationTestAccess::isRankingObjectives(*population));

    const ScopedParetoSelection pareto;
    REQUIRE(PopulationTestAccess::isRankingObjectives(*population));
}

TEST_CASE("Population::calculateAdjustedFitness divides the selection fitness by the species size", "[ParetoSelection]")
{
    // Three fronts: {(0.9,0.9)}, {(0.5,0.5),(0.6,0.4)}, {(0.1,0.1)}.
    const std::vector<std::vector<double>> points{ { 0.9, 0.9 }, { 0.5, 0.5 }, { 0.6, 0.4 }, { 0.1, 0.1 } };

    SECTION("scalar mode: fitness / species size")
    {
        SelectionConstants::reset();
        const auto population = populationAt(points);
        PopulationTestAccess::speciate(*population);
        for (const auto& solution : population->getSolutions())
        {
            const auto speciesSize = static_cast<double>(PopulationTestAccess::speciesOf(*population, solution)->size());
            REQUIRE(solution->getParameters().adjustedFitness == Catch::Approx(solution->getFitness() / speciesSize));
        }
    }
    SECTION("Pareto mode: (fronts - rank) / fronts / species size")
    {
        const ScopedParetoSelection pareto;
        const auto population = populationAt(points);
        PopulationTestAccess::rankObjectives(*population);
        PopulationTestAccess::speciate(*population);
        for (const auto& solution : population->getSolutions())
        {
            const auto parameters = solution->getParameters();
            const double expectedSelectionFitness = (3.0 - parameters.paretoRank) / 3.0;
            REQUIRE(solution->getSelectionFitness() == Catch::Approx(expectedSelectionFitness));
            const auto speciesSize = static_cast<double>(PopulationTestAccess::speciesOf(*population, solution)->size());
            REQUIRE(parameters.adjustedFitness == Catch::Approx(expectedSelectionFitness / speciesSize));
        }
    }
}

TEST_CASE("Population::preserveGlobalBestSolution evicts the worst solution by the comparator", "[ParetoSelection]")
{
    const auto run = [](const std::vector<std::vector<double>>& points, const std::vector<double>& fitnesses)
    {
        auto population = populationAt(points);
        auto& solutions = PopulationTestAccess::solutions(*population);
        for (size_t i = 0; i < solutions.size(); ++i)
        {
            solutions[i] = evaluatedAt(points[i], fitnesses[i]);
        }
        if (SelectionConstants::mode == SelectionMode::Pareto)
        {
            PopulationTestAccess::rankObjectives(*population);
        }
        PopulationTestAccess::speciate(*population);
        const auto survivors = solutions;
        PopulationTestAccess::setPreviousBestSolution(*population, evaluatedAt({ 0.5, 0.5 }, 0.5));
        PopulationTestAccess::preserveGlobalBestSolution(*population);
        return std::pair{ survivors, population->getSolutions() };
    };
    // Fitness disagrees with dominance: (0.1,0.1) is dominated by both others but has the top fitness.
    const std::vector<std::vector<double>> points{ { 0.9, 0.9 }, { 0.1, 0.1 }, { 0.5, 0.6 } };
    const std::vector<double> fitnesses{ 0.5, 0.9, 0.2 };
    const auto contains = [](const std::vector<SolutionPtr>& solutions, const SolutionPtr& solution)
    { return std::ranges::find(solutions, solution) != solutions.end(); };

    SECTION("scalar mode evicts the lowest fitness")
    {
        SelectionConstants::reset();
        const auto [before, after] = run(points, fitnesses);
        REQUIRE(after.size() == before.size());
        REQUIRE_FALSE(contains(after, before[2]));
        REQUIRE(contains(after, before[1]));
    }
    SECTION("Pareto mode evicts the last front")
    {
        const ScopedParetoSelection pareto;
        const auto [before, after] = run(points, fitnesses);
        REQUIRE(after.size() == before.size());
        REQUIRE_FALSE(contains(after, before[1]));
        REQUIRE(contains(after, before[2]));
    }
}

TEST_CASE("Population picks and sorts the best species by champion comparator", "[ParetoSelection]")
{
    const auto population = populationAt({ { 0.5, 0.5 } });
    auto& speciesList = PopulationTestAccess::speciesList(*population);
    const auto frontZeroChampion = rankedAt({ 0.9, 0.1 }, 0.1, 0, infinity);
    const auto fitterChampion = rankedAt({ 0.8, 0.05 }, 0.9, 1, infinity);
    for (const auto& champion : { fitterChampion, frontZeroChampion })
    {
        auto species = std::make_shared<Species>();
        species->addSolution(champion);
        species->assignChampion(true);
        speciesList.push_back(species);
    }

    SECTION("scalar mode: the fitter champion")
    {
        SelectionConstants::reset();
        REQUIRE(PopulationTestAccess::bestActiveSpecies(*population)->getChampion() == fitterChampion);
        PopulationTestAccess::sortSpeciesListByChampionFitness(*population);
        REQUIRE(speciesList.front()->getChampion() == fitterChampion);
    }
    SECTION("Pareto mode: the champion on the lower front")
    {
        const ScopedParetoSelection pareto;
        REQUIRE(PopulationTestAccess::bestActiveSpecies(*population)->getChampion() == frontZeroChampion);
        PopulationTestAccess::sortSpeciesListByChampionFitness(*population);
        REQUIRE(speciesList.front()->getChampion() == frontZeroChampion);
    }
}

// --- improvement signals (section 2, rows 2 and 5) ---------------------------------

TEST_CASE("Pareto-mode population improvement is the archive accepting a point", "[ParetoSelection]")
{
    const ScopedParetoSelection pareto;
    const auto population = populationAt({ { 0.8, 0.2 }, { 0.2, 0.8 }, { 0.1, 0.1 } });

    PopulationTestAccess::rankObjectives(*population);
    REQUIRE(PopulationTestAccess::hasParetoFrontImproved(*population));
    REQUIRE(PopulationTestAccess::archiveSize(*population) == 2);

    // The same front again: the archive rejects both points as equal.
    PopulationTestAccess::rankObjectives(*population);
    REQUIRE_FALSE(PopulationTestAccess::hasParetoFrontImproved(*population));

    // A point that extends the front.
    moveSolutionTo(*population, 2, { 0.5, 0.6 });
    PopulationTestAccess::rankObjectives(*population);
    REQUIRE(PopulationTestAccess::hasParetoFrontImproved(*population));
}

TEST_CASE("Pareto-mode stagnation counts archive improvements, not best-fitness gains", "[ParetoSelection]")
{
    const auto countStagnantCalls = [](const bool pareto)
    {
        if (pareto)
        {
            SelectionConstants::mode = SelectionMode::Pareto;
        }
        const auto population = populationAt({ { 0.8, 0.2 }, { 0.2, 0.8 } });
        PopulationTestAccess::rankObjectives(*population);
        // Every generation the scalar best rises, while the front stays where it is.
        for (int generation = 1; generation <= PopulationConstants::generationsWithoutImprovementThresholdInPopulation; ++generation)
        {
            PopulationTestAccess::rankObjectives(*population);
            PopulationTestAccess::setBestSolution(*population, evaluatedAt({ 0.5 }, 0.1 * generation));
            if (!PopulationTestAccess::hasFitnessImprovedOverTheLastGenerations(*population))
            {
                SelectionConstants::reset();
                return generation;
            }
        }
        SelectionConstants::reset();
        return 0;
    };

    REQUIRE(countStagnantCalls(false) == 0);
    REQUIRE(countStagnantCalls(true) == PopulationConstants::generationsWithoutImprovementThresholdInPopulation);
}

TEST_CASE("Pareto-mode improvement falls back to the lowest violation while nothing is feasible", "[ParetoSelection]")
{
    // Floor 0.9: every point below is infeasible. (0.5,0.5) falls short by 0.8; (0.2,0.2) by 1.4.
    const ScopedParetoSelection pareto(0.9, 0.05);
    const auto population = populationAt({ { 0.5, 0.5 }, { 0.2, 0.2 } });

    PopulationTestAccess::rankObjectives(*population);
    REQUIRE(PopulationTestAccess::hasParetoFrontImproved(*population));

    PopulationTestAccess::rankObjectives(*population);
    REQUIRE_FALSE(PopulationTestAccess::hasParetoFrontImproved(*population));

    // Shortfall 0.78: lower, but by less than epsilon.
    moveSolutionTo(*population, 1, { 0.51, 0.51 });
    PopulationTestAccess::rankObjectives(*population);
    REQUIRE_FALSE(PopulationTestAccess::hasParetoFrontImproved(*population));

    // Shortfall 0.6.
    moveSolutionTo(*population, 1, { 0.6, 0.6 });
    PopulationTestAccess::rankObjectives(*population);
    REQUIRE(PopulationTestAccess::hasParetoFrontImproved(*population));
    REQUIRE(PopulationTestAccess::archiveSize(*population) == 0);
}

TEST_CASE("Pareto-mode species improvement follows the accepted solutions' current species", "[ParetoSelection]")
{
    const ScopedParetoSelection pareto;
    const auto population = populationAt({ { 0.9, 0.9 }, { 0.1, 0.1 } });
    const auto solutions = population->getSolutions();
    const auto accepted = solutions[0];
    const auto dominated = solutions[1];
    Species acceptedSpecies;
    Species otherSpecies;
    // The archive entry would carry this stale id; the signal must not trust it.
    accepted->setSpeciesId(otherSpecies.getId());

    PopulationTestAccess::rankObjectives(*population);
    acceptedSpecies.addSolution(accepted);
    otherSpecies.addSolution(dominated);

    REQUIRE(PopulationTestAccess::hasSpeciesImprovedOnTheFront(*population, acceptedSpecies));
    REQUIRE_FALSE(PopulationTestAccess::hasSpeciesImprovedOnTheFront(*population, otherSpecies));
}

TEST_CASE("Pareto-mode species improvement falls back to holding the new lowest violation", "[ParetoSelection]")
{
    const ScopedParetoSelection pareto(0.9);
    const auto population = populationAt({ { 0.5, 0.5 }, { 0.2, 0.2 } });
    const auto solutions = population->getSolutions();
    Species leastViolating;
    Species mostViolating;
    leastViolating.addSolution(solutions[0]);
    mostViolating.addSolution(solutions[1]);

    PopulationTestAccess::rankObjectives(*population);
    REQUIRE(PopulationTestAccess::hasSpeciesImprovedOnTheFront(*population, leastViolating));
    REQUIRE_FALSE(PopulationTestAccess::hasSpeciesImprovedOnTheFront(*population, mostViolating));

    PopulationTestAccess::rankObjectives(*population);
    REQUIRE_FALSE(PopulationTestAccess::hasSpeciesImprovedOnTheFront(*population, leastViolating));
}

// --- integration ------------------------------------------------------------------

TEST_CASE("Population::evolve runs Pareto selection on a conflicting-objective stub", "[Population]")
{
    const ScopedParetoSelection pareto(0.0, 0.01);
    resetGlobalState();
    constexpr int populationSize = 20;
    constexpr int generations = 5;
    const PopulationParameters parameters(populationSize, generations, 1.1);
    const std::string solutionName = "ObjectiveStub";
    const auto parentDirectory = paths::dataRoot() / "data" / solutionName;
    std::unordered_set<std::string> preExisting;
    if (std::filesystem::exists(parentDirectory))
    {
        for (const auto& entry : std::filesystem::directory_iterator(parentDirectory))
        {
            preExisting.insert(entry.path().filename().string());
        }
    }

    Population population(parameters, std::make_shared<ObjectiveStubSolution>(makeTopology(1, 1)));
    population.setValidationPolicy(ValidationPolicy::Throw);
    population.initialize();
    REQUIRE_NOTHROW(population.evolve());
    REQUIRE(population.getValidationReport().clean());
    REQUIRE(population.getCurrentGeneration() == generations);

    std::string runDirectory;
    for (const auto& entry : std::filesystem::directory_iterator(parentDirectory))
    {
        if (!preExisting.contains(entry.path().filename().string()))
        {
            runDirectory = entry.path().generic_string() + "/";
        }
    }
    REQUIRE(!runDirectory.empty());

    std::ifstream objectivesFile(runDirectory + "objectives.jsonl");
    std::vector<nlohmann::json> records;
    for (std::string line; std::getline(objectivesFile, line);)
    {
        if (!line.empty())
        {
            records.push_back(nlohmann::json::parse(line));
        }
    }
    REQUIRE(records.size() == generations);
    for (const auto& record : records)
    {
        REQUIRE(record.at("mode") == "pareto");
        const auto& individuals = record.at("individuals");
        REQUIRE(individuals.size() == populationSize);
        for (const auto& dominator : individuals)
        {
            for (const auto& dominated : individuals)
            {
                const RankedPoint a{ dominator.at("objectives").get<std::vector<double>>(), dominator.at("violation").get<double>() };
                const RankedPoint b{ dominated.at("objectives").get<std::vector<double>>(), dominated.at("violation").get<double>() };
                if (constrainedDominates(a, b, 0.01))
                {
                    REQUIRE(dominator.at("rank").get<int>() < dominated.at("rank").get<int>());
                }
            }
        }
    }
    const auto finalArchiveSize = records.back().at("archive").at("size").get<size_t>();
    REQUIRE(finalArchiveSize > 0);

    // End of a Pareto run: the archive, and the phenotypes of its members still alive.
    std::ifstream archiveFile(runDirectory + "pareto_archive.json");
    REQUIRE(archiveFile.is_open());
    const auto archive = nlohmann::json::parse(archiveFile);
    REQUIRE(archive.at("mode") == "pareto");
    REQUIRE(archive.at("members").size() == finalArchiveSize);
    size_t alive = 0;
    for (const auto& member : archive.at("members"))
    {
        REQUIRE(member.at("objectives").size() == 2);
        REQUIRE(member.at("partialFitness").size() == 2);
        REQUIRE(member.contains("id"));
        REQUIRE(member.contains("generationFound"));
        REQUIRE(member.contains("fitness"));
        alive += member.at("inFinalPopulation").get<bool>() ? 1 : 0;
    }
    const auto frontDirectory = runDirectory + "pareto_front/";
    REQUIRE(std::filesystem::is_directory(frontDirectory));
    const auto savedPhenotypes = static_cast<size_t>(std::distance(
        std::filesystem::directory_iterator(frontDirectory), std::filesystem::directory_iterator{}));
    REQUIRE(savedPhenotypes == alive);

    objectivesFile.close();
    archiveFile.close();
    std::filesystem::remove_all(runDirectory);
}
