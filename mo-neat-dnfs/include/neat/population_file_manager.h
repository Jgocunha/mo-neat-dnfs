#pragma once

#include <memory>
#include <string>

#include <dnf_composer/simulation/simulation_file_manager.h>

namespace neat_dnfs
{
	class Population;
	class Solution;

	/// @brief Handles all on-disk persistence for a Population run.
	///
	/// Holds a back-reference to the Population it serves and reads its state through
	/// const getters. A Population that is constructed without a file manager performs
	/// no file I/O at all (used by the unit tests). Each @c save* method honours the
	/// corresponding @c PopulationConstants::saveXxx compile-time flag internally.
	class PopulationFileManager
	{
	public:
		explicit PopulationFileManager(const Population& population);

		/// @brief Builds the timestamped output directory for this run. Call once at startup.
		void setFileDirectory();

		/// @brief Writes the per-generation overview line. Called before generation/age updates.
		void saveOverviewForGeneration() const;
		/// @brief Writes the remaining per-generation artifacts. Called after generation/age updates.
		void savePerGenerationData() const;
		/// @brief Writes end-of-run artifacts (final solutions, timestamps, champions, and in
		/// Pareto mode the archive).
		void saveEndOfRunData() const;

#ifdef NEAT_DNFS_PROFILE
		/// @brief Appends the current generation's profiler buckets as one row of
		/// profile.csv in the run directory, creating the file with a header row
		/// the first time it is called. Compiled only when @c NEAT_DNFS_PROFILE is on.
		/// @details Columns are a fixed, explicit set (evaluate, speciate, upkeep,
		/// reproduceAndSelect, save), not whatever tools::profiler::snapshot()
		/// happens to hold that generation -- that keeps every row's columns
		/// aligned with the header even on a generation where a phase's bucket is
		/// absent. The save column times only file I/O and is measured *inside*
		/// the upkeep scope in Population::evolve(), so upkeep's own total is
		/// inclusive of save rather than disjoint from it; save is still reported
		/// separately since it is usually the more actionable of the two.
		void saveProfileForGeneration() const;
#endif

	private:
		void saveAllSolutionsWithFitnessAbove(double fitness) const;
		/// @brief Saves @p solution's phenotype elements (and coupling weights) as JSON.
		/// @param solution The solution to save.
		/// @param directoryPath Directory the elements are written to.
		void saveSolutionPhenotype(const std::shared_ptr<Solution>& solution, const std::string& directoryPath) const;
		/// @brief Writes pareto_archive.json at the end of a Pareto-mode run: the selection
		/// settings, then every archive member's id, generation found, objectives, partials,
		/// fitness and whether it is still in the final population. When
		/// PopulationConstants::saveParetoFront is on, also saves the phenotypes of those
		/// still alive under pareto_front/last_generation/. A member that died earlier is
		/// on disk under pareto_front/gen <generationFound + 1>/ instead.
		void saveParetoArchive() const;
		/// @brief Saves the phenotype of every archive member in the current generation's
		/// population under pareto_front/gen <generation>/, numbered like solutions/gen M:
		/// one more than the generation objectives.jsonl and generationFound use. Every
		/// member is in the population of the generation that found it, so each one is
		/// saved at least there. Writes nothing (no directory) when none is present.
		void saveParetoFrontForGeneration() const;
		/// @brief Saves the phenotype of every archive member still in the population.
		/// @param directoryPath Directory the phenotypes are written to; created only if
		/// at least one member is present.
		void saveArchiveMembersInPopulation(const std::string& directoryPath) const;
		void saveChampions() const;
		void saveTimestampsAndDuration() const;
		void saveRunMetadata() const;
		void saveAllSolutionsPerGeneration() const;
		void savePerGenerationOverview() const;
		/// @brief Appends one JSON object for the current generation to overview.jsonl,
		/// alongside (never replacing) the prose per_generation_overview.txt.
		void savePerGenerationOverviewJson() const;
		/// @brief Appends one JSON object for the current generation to objectives.jsonl:
		/// the selection settings, every solution's partials, objectives, Pareto rank,
		/// crowding distance and constraint violation, and the archive's size and
		/// this generation's accepted ids. An infinite crowding distance (a boundary
		/// point) is written as null, since JSON has no infinity.
		void saveObjectivesForGeneration() const;
		void saveBestSolutionOfEachGeneration() const;
		void saveChampionsOfEachGeneration() const;
		void savePerGenerationStatistics() const;
		void savePerGenerationSpecies() const;

		const Population* population;
		std::string fileDirectory;
	};
}
