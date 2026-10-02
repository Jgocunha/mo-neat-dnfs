#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <map>

#include "constants.h"
#include "neat_tools/ablation_presets.h"
#include "neat_tools/config_loader.h"
#include "neat_tools/solution_registry.h"
#include "solutions/xor.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using nlohmann::json;

namespace
{
    // Every test here reloads the global config from a file it wrote, so the
    // reference config is restored afterwards for the rest of the binary.
    struct RestoreReferenceConfig
    {
        RestoreReferenceConfig() = default;
        ~RestoreReferenceConfig()
        {
            ConfigLoader::loadGlobalConfig(ConfigLoader::defaultGlobalConfigPath());
        }
        RestoreReferenceConfig(const RestoreReferenceConfig&) = delete;
        RestoreReferenceConfig& operator=(const RestoreReferenceConfig&) = delete;
        RestoreReferenceConfig(RestoreReferenceConfig&&) = delete;
        RestoreReferenceConfig& operator=(RestoreReferenceConfig&&) = delete;
    };

    // The shipped reference config, as the starting point each test modifies.
    json referenceConfig()
    {
        return ConfigLoader::loadJsonFile(ConfigLoader::defaultGlobalConfigPath());
    }

    // Writes under <git root>/.claude/temp/ (gitignored); distinct file names per test
    // keep parallel ctest processes from overwriting each other's config.
    std::string writeTempConfig(const json& config, const std::string& fileName)
    {
        const auto directory = std::filesystem::path(PROJECT_DIR) / ".." / ".claude" / "temp" / "selection-config";
        std::filesystem::create_directories(directory);
        const auto path = directory / fileName;
        std::ofstream(path) << config.dump(2);
        return path.generic_string();
    }

    // Loads the reference config with its SelectionConstants block replaced by block.
    void loadWithSelectionBlock(const json& block, const std::string& fileName)
    {
        auto config = referenceConfig();
        config["SelectionConstants"] = block;
        ConfigLoader::loadGlobalConfig(writeTempConfig(config, fileName));
    }

    // Moves every selection field off its default, so a test can prove a load reset it.
    void setNonDefaultSelection()
    {
        SelectionConstants::mode = SelectionMode::Pareto;
        SelectionConstants::objectiveGroups = { { 0 }, { 1 } };
        SelectionConstants::dominanceEpsilon = 0.2;
        SelectionConstants::feasibilityFloor = 0.3;
        SelectionConstants::archiveCapacity = 5;
        SelectionConstants::archiveEpsilon = 0.25;
        SelectionConstants::violationEpsilon = 0.35;
    }

    // Asserts every selection field holds its compiled-in default.
    void requireDefaultSelection()
    {
        REQUIRE(SelectionConstants::mode == SelectionMode::Scalar);
        REQUIRE(SelectionConstants::objectiveGroups.empty());
        REQUIRE(SelectionConstants::dominanceEpsilon == 0.0);
        REQUIRE(SelectionConstants::feasibilityFloor == 0.0);
        REQUIRE(SelectionConstants::archiveCapacity == 100);
        REQUIRE(SelectionConstants::archiveEpsilon == 0.0);
        REQUIRE(SelectionConstants::violationEpsilon == 0.0);
    }

    // Restores the default (empty) grouping even when construction throws.
    struct ScopedObjectiveGroups
    {
        explicit ScopedObjectiveGroups(std::vector<std::vector<size_t>> groups)
        {
            SelectionConstants::objectiveGroups = std::move(groups);
        }
        ~ScopedObjectiveGroups()
        {
            SelectionConstants::reset();
        }
        ScopedObjectiveGroups(const ScopedObjectiveGroups&) = delete;
        ScopedObjectiveGroups& operator=(const ScopedObjectiveGroups&) = delete;
        ScopedObjectiveGroups(ScopedObjectiveGroups&&) = delete;
        ScopedObjectiveGroups& operator=(ScopedObjectiveGroups&&) = delete;
    };
}

TEST_CASE("SelectionConstants::reset restores every default", "[SelectionConfig]")
{
    setNonDefaultSelection();

    SelectionConstants::reset();

    requireDefaultSelection();
}

TEST_CASE("The reference config holds the default selection block", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    setNonDefaultSelection();
    PopulationConstants::saveObjectives = false;

    ConfigLoader::loadGlobalConfig(ConfigLoader::defaultGlobalConfigPath());

    requireDefaultSelection();
    REQUIRE(PopulationConstants::saveObjectives);
    const auto reference = referenceConfig();
    REQUIRE(reference.contains("SelectionConstants"));
    REQUIRE(reference.at("PopulationConstants").contains("saveObjectives"));
}

TEST_CASE("ConfigLoader parses a full SelectionConstants block", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    loadWithSelectionBlock({
        { "mode", "pareto" },
        { "objectiveGroups", { { 0, 2 }, { 1, 3 } } },
        { "dominanceEpsilon", 0.05 },
        { "feasibilityFloor", 0.2 },
        { "archiveCapacity", 7 },
    }, "full-block.json");

    REQUIRE(SelectionConstants::mode == SelectionMode::Pareto);
    REQUIRE(SelectionConstants::objectiveGroups == std::vector<std::vector<size_t>>{ { 0, 2 }, { 1, 3 } });
    REQUIRE(SelectionConstants::dominanceEpsilon == Catch::Approx(0.05));
    REQUIRE(SelectionConstants::feasibilityFloor == Catch::Approx(0.2));
    REQUIRE(SelectionConstants::archiveCapacity == 7);
}

TEST_CASE("ConfigLoader rejects an unknown selection mode", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    REQUIRE_THROWS_WITH(loadWithSelectionBlock({ { "mode", "nsga3" } }, "unknown-mode.json"),
        Catch::Matchers::ContainsSubstring("nsga3"));
}

TEST_CASE("ConfigLoader accepts a config with no SelectionConstants block", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    auto config = referenceConfig();
    config.erase("SelectionConstants");
    const auto path = writeTempConfig(config, "no-block.json");
    setNonDefaultSelection();

    REQUIRE_NOTHROW(ConfigLoader::loadGlobalConfig(path));

    requireDefaultSelection();
}

TEST_CASE("ConfigLoader fills the fields a partial SelectionConstants block omits with defaults", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    setNonDefaultSelection();

    loadWithSelectionBlock({ { "feasibilityFloor", 0.1 } }, "partial-block.json");

    REQUIRE(SelectionConstants::feasibilityFloor == Catch::Approx(0.1));
    REQUIRE(SelectionConstants::mode == SelectionMode::Scalar);
    REQUIRE(SelectionConstants::objectiveGroups.empty());
    REQUIRE(SelectionConstants::dominanceEpsilon == 0.0);
    REQUIRE(SelectionConstants::archiveCapacity == 100);
}

TEST_CASE("ConfigLoader rejects a mistyped key inside SelectionConstants", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    REQUIRE_THROWS_WITH(loadWithSelectionBlock({ { "dominanceEpsilion", 0.01 } }, "mistyped-key.json"),
        Catch::Matchers::ContainsSubstring("dominanceEpsilion"));
}

TEST_CASE("ConfigLoader rejects a SelectionConstants value that is not an object", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    const std::vector<json> invalidBlocks{ json(nullptr), json::array(), json("pareto"), json(1) };
    for (const auto& block : invalidBlocks)
    {
        INFO(block.dump());
        REQUIRE_THROWS_WITH(loadWithSelectionBlock(block, "not-an-object.json"),
            Catch::Matchers::ContainsSubstring("SelectionConstants must be an object"));
    }
}

TEST_CASE("ConfigLoader rejects out-of-range selection values", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    const std::vector<json> invalidBlocks{
        { { "dominanceEpsilon", -0.01 } },
        { { "dominanceEpsilon", 0.5 } },
        { { "feasibilityFloor", -0.01 } },
        { { "feasibilityFloor", 1.0 } },
        { { "archiveCapacity", 0 } },
        { { "archiveCapacity", -3 } },
    };
    for (const auto& block : invalidBlocks)
    {
        INFO(block.dump());
        REQUIRE_THROWS_WITH(loadWithSelectionBlock(block, "out-of-range.json"),
            Catch::Matchers::ContainsSubstring(block.begin().key()));
    }
}

TEST_CASE("ConfigLoader defaults saveObjectives to true when the key is absent", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    auto config = referenceConfig();
    config["PopulationConstants"].erase("saveObjectives");
    const auto path = writeTempConfig(config, "no-save-objectives.json");
    PopulationConstants::saveObjectives = false;

    REQUIRE_NOTHROW(ConfigLoader::loadGlobalConfig(path));

    REQUIRE(PopulationConstants::saveObjectives);
}

TEST_CASE("An ablation preset can switch on Pareto selection", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    ConfigLoader::loadGlobalConfig(ConfigLoader::defaultGlobalConfigPath());
    const json preset{
        { "AblationConstants", { { "label", " Pareto" } } },
        { "SelectionConstants", { { "mode", "pareto" }, { "feasibilityFloor", 0.1 } } },
    };

    ConfigLoader::applyAblation(writeTempConfig(preset, "pareto-preset.json"));

    REQUIRE(SelectionConstants::mode == SelectionMode::Pareto);
    REQUIRE(SelectionConstants::feasibilityFloor == Catch::Approx(0.1));
    REQUIRE(AblationConstants::label == " Pareto");
}

TEST_CASE("Solution construction rejects objective groups that do not partition the partials", "[SelectionConfig]")
{
    resetGlobalState();

    const std::vector<std::pair<std::string, std::vector<std::vector<size_t>>>> invalidGroupings{
        { "overlap", { { 0, 1 }, { 1, 2, 3 } } },
        { "gap", { { 0, 1 }, { 3 } } },
        { "out of range", { { 0, 1 }, { 2, 3, 4 } } },
        { "empty group", { { 0, 1, 2, 3 }, {} } },
    };
    for (const auto& [description, groups] : invalidGroupings)
    {
        INFO(description);
        const ScopedObjectiveGroups scoped(groups);
        REQUIRE_THROWS_WITH(XOR(makeTopology(2, 1)), Catch::Matchers::ContainsSubstring("xor"));
    }
}

TEST_CASE("Solution construction accepts objective groups that partition the partials", "[SelectionConfig]")
{
    resetGlobalState();
    const ScopedObjectiveGroups scoped({ { 3, 0 }, { 2, 1 } });

    REQUIRE_NOTHROW(XOR(makeTopology(2, 1)));
}

TEST_CASE("Solution::evaluate derives objectives from the partial fitnesses", "[SelectionConfig]")
{
    resetGlobalState();
    const std::vector<double> partials{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    SECTION("no groups: one objective per partial")
    {
        const ScopedObjectiveGroups scoped({});
        FixedPartialsSolution solution(makeTopology(1, 1), partials);

        solution.evaluate();

        REQUIRE(solution.getParameters().objectives == partials);
    }

    SECTION("grouped: the weighted mean of each group under the task's weights")
    {
        // config/solutions/and.json weights: 0.10 0.20 0.10 0.20 0.25 0.05 0.05 0.05
        const ScopedObjectiveGroups scoped({ { 0, 2 }, { 1, 3, 4 }, { 5, 6, 7 } });
        FixedPartialsSolution solution(makeTopology(1, 1), partials);

        solution.evaluate();

        const auto objectives = solution.getParameters().objectives;
        REQUIRE(objectives.size() == 3);
        REQUIRE(objectives[0] == Catch::Approx((0.10 * 0.1 + 0.10 * 0.3) / 0.20));
        REQUIRE(objectives[1] == Catch::Approx((0.20 * 0.2 + 0.20 * 0.4 + 0.25 * 0.5) / 0.65));
        REQUIRE(objectives[2] == Catch::Approx((0.6 + 0.7 + 0.8) / 3.0));
    }
}

TEST_CASE("Every task config groups its partials into the planned objectives", "[SelectionConfig]")
{
    // .claude/notes/MOO/PLAN.md section 3.6: each grouping was checked against a real run.
    const std::map<std::string, std::vector<std::vector<size_t>>> plannedGroups{
        { "and", { { 0, 2 }, { 1, 3, 4 }, { 5, 6, 7 } } },
        { "xor", { { 0, 1, 2 }, { 3 } } },
        { "detection-instability", { { 0, 1 }, { 2, 3 } } },
        { "memory-instability", { { 0, 2 }, { 1, 3 } } },
        { "selection-instability", { { 0, 1 }, { 2, 3 } } },
        { "memory-trace", { { 0, 2 }, { 1, 3, 4 }, { 5, 6, 7 } } },
        { "dmts", { { 0, 2, 4 }, { 1, 3, 5 } } },
        { "ior", { { 0, 2 }, { 1, 3, 4 } } },
        // Added by the neat-dnfs #120 sync; no grouping has been chosen yet, so
        // each of its 7 partials is its own objective.
        { "hri-packaging", {} },
    };
    REQUIRE(plannedGroups.size() == taskEntries().size());

    for (const auto& task : taskEntries())
    {
        const std::string slug(task.slug);
        INFO(slug);
        resetGlobalState();
        const ScopedTaskConfig taskConfig{ slug };

        REQUIRE(SelectionConstants::objectiveGroups == plannedGroups.at(slug));
        REQUIRE_NOTHROW(task.makeFromTopology(defaultTopologyFor(task)));
    }
}

TEST_CASE("ScopedTaskConfig restores the default selection settings", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    resetGlobalState();
    {
        const ScopedTaskConfig taskConfig{ "and" };
    }

    requireDefaultSelection();
    REQUIRE_NOTHROW(XOR(makeTopology(2, 1)));

    SECTION("including for an ablation applied afterwards")
    {
        REQUIRE(AblationPresets::applyByName("no-crossover"));

        requireDefaultSelection();
        REQUIRE_NOTHROW(XOR(makeTopology(2, 1)));
    }
}

TEST_CASE("The pareto-selection presets switch on Pareto selection", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    SECTION("pareto-selection: epsilon 0.01, floor 0.1")
    {
        REQUIRE(AblationPresets::applyByName("pareto-selection"));

        REQUIRE(AblationConstants::label == " Pareto");
        REQUIRE(SelectionConstants::feasibilityFloor == Catch::Approx(0.1));
    }

    SECTION("pareto-selection-no-floor: epsilon 0.01, floor off")
    {
        REQUIRE(AblationPresets::applyByName("pareto-selection-no-floor"));

        REQUIRE(AblationConstants::label == " Pareto NoFloor");
        REQUIRE(SelectionConstants::feasibilityFloor == 0.0);
    }

    REQUIRE(SelectionConstants::mode == SelectionMode::Pareto);
    REQUIRE(SelectionConstants::dominanceEpsilon == Catch::Approx(0.01));
    REQUIRE(SelectionConstants::archiveCapacity == 100);
}

TEST_CASE("A pareto-selection preset keeps the task's objective groups", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    ConfigLoader::loadConfig(ConfigLoader::defaultGlobalConfigPath(), "and");

    REQUIRE(AblationPresets::applyByName("pareto-selection"));

    REQUIRE(SelectionConstants::objectiveGroups == std::vector<std::vector<size_t>>{ { 0, 2 }, { 1, 3, 4 }, { 5, 6, 7 } });
}

TEST_CASE("ConfigLoader defaults saveParetoFront to true when the key is absent", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    auto config = referenceConfig();
    config["PopulationConstants"].erase("saveParetoFront");
    const auto path = writeTempConfig(config, "no-save-pareto-front.json");
    PopulationConstants::saveParetoFront = false;

    REQUIRE_NOTHROW(ConfigLoader::loadGlobalConfig(path));

    REQUIRE(PopulationConstants::saveParetoFront);
}

TEST_CASE("ConfigLoader reads saveParetoFront from PopulationConstants", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;
    auto config = referenceConfig();
    config["PopulationConstants"]["saveParetoFront"] = false;

    ConfigLoader::loadGlobalConfig(writeTempConfig(config, "save-pareto-front-off.json"));

    REQUIRE_FALSE(PopulationConstants::saveParetoFront);
}

TEST_CASE("ConfigLoader reads archiveEpsilon and violationEpsilon", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    loadWithSelectionBlock({
        { "dominanceEpsilon", 0.0 },
        { "archiveEpsilon", 0.02 },
        { "violationEpsilon", 0.03 },
    }, "tolerances.json");

    REQUIRE(SelectionConstants::dominanceEpsilon == 0.0);
    REQUIRE(SelectionConstants::archiveEpsilon == Catch::Approx(0.02));
    REQUIRE(SelectionConstants::violationEpsilon == Catch::Approx(0.03));
}

TEST_CASE("ConfigLoader defaults archiveEpsilon to dominanceEpsilon and violationEpsilon to 0", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    loadWithSelectionBlock({ { "dominanceEpsilon", 0.04 } }, "tolerance-defaults.json");

    REQUIRE(SelectionConstants::archiveEpsilon == Catch::Approx(0.04));
    REQUIRE(SelectionConstants::violationEpsilon == 0.0);
}

TEST_CASE("ConfigLoader rejects out-of-range tolerances", "[SelectionConfig]")
{
    const RestoreReferenceConfig restore;

    const std::vector<json> invalidBlocks{
        { { "archiveEpsilon", -0.01 } },
        { { "archiveEpsilon", 0.5 } },
        { { "violationEpsilon", -0.01 } },
        { { "violationEpsilon", 1.0 } },
    };
    for (const auto& block : invalidBlocks)
    {
        INFO(block.dump());
        REQUIRE_THROWS_WITH(loadWithSelectionBlock(block, "out-of-range-tolerance.json"),
            Catch::Matchers::ContainsSubstring(block.begin().key()));
    }
}
