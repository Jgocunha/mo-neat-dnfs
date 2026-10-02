#pragma once

#include "neat/population.h"

namespace neat_dnfs::test {

// Grants tests direct, deterministic access to Population's private
// speciation/reproduction internals (issue #59: extinct species were never
// erased from Population::speciesList, and assignToSpecies() re-randomized
// an existing species' representative on every insertion during a single
// assignment pass). Reproducing either defect through the public evolve()
// API alone would mean waiting on stochastic genome compatibility and
// fitness dynamics to happen to trigger them -- this lets the regression
// tests drive the exact internal state instead.
class PopulationTestAccess
{
public:
    static void speciate(Population& population) { population.speciate(); }
    static void reproduceAndSelect(Population& population) { population.reproduceAndSelect(); }
    static std::vector<std::shared_ptr<Species>>& speciesList(Population& population) { return population.speciesList; }
    static void setBestSolution(Population& population, const SolutionPtr& solution) { population.bestSolution = solution; }

    // Pareto selection (Phase 5): drive one step at a time and read the signals.
    static std::vector<SolutionPtr>& solutions(Population& population) { return population.solutions; }
    static void evaluate(const Population& population) { population.evaluate(); }
    static void rankObjectives(Population& population) { population.rankObjectives(); }
    static bool isRankingObjectives(const Population& population) { return population.isRankingObjectives(); }
    static bool hasParetoFrontImproved(const Population& population) { return population.hasParetoFrontImproved(); }
    static bool offerSpeciesToItsFront(Population& population, const Species& species) { return population.offerSpeciesToItsFront(species); }
    static bool hasFitnessImprovedOverTheLastGenerations(Population& population) { return population.hasFitnessImprovedOverTheLastGenerations(); }
    static size_t archiveSize(const Population& population) { return population.paretoArchive.size(); }
    static std::shared_ptr<Species> speciesOf(Population& population, const SolutionPtr& solution) { return population.findSpecies(solution); }
    static void setPreviousBestSolution(Population& population, const SolutionPtr& solution) { population.previousBestSolution = solution; }
    static void preserveGlobalBestSolution(Population& population) { population.preserveGlobalBestSolution(); }
    static std::shared_ptr<Species> bestActiveSpecies(const Population& population) { return population.getBestActiveSpecies(); }
    static void sortSpeciesListByChampionFitness(Population& population) { population.sortSpeciesListByChampionFitness(); }
};

} // namespace neat_dnfs::test
