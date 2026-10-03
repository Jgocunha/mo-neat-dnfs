#pragma once

#include <array>
#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "genome.h"
#include "pareto.h"
#include "solution.h"
#include "species.h"

namespace neat_dnfs
{
	class PopulationFileManager;

	namespace test { class PopulationTestAccess; }

	/// @brief Identifies which internal invariant check reported a violation.
	enum class ValidationCheck
	{
		PopulationSize,
		UniqueSolutions,
		Elitism,
		UniqueGenesInGenomes,
		UniqueKernelAndNeuralFieldPtrs,
		SpeciesHaveUniqueRepresentative,
		AssignmentIntoSpecies,
		Count
	};

	/// @brief How a Population reacts to an invariant violation.
	/// Log (the production default) only records/prints it; Throw raises
	/// ValidationError, which is what the test binary opts into via
	/// Population::setDefaultValidationPolicy so violations fail tests loudly
	/// without changing production behaviour.
	enum class ValidationPolicy
	{
		Log,
		Throw
	};

	/// @brief Accumulates invariant-check violations observed during a run.
	/// Messages are capped (not one per violation) since an O(n^2) check at a
	/// large population size can otherwise emit far more strings than anyone
	/// will read; counts remain exact regardless of the cap.
	struct ValidationReport
	{
		static constexpr size_t maxRetainedMessages = 32;

		std::array<int, static_cast<size_t>(ValidationCheck::Count)> counts{};
		std::vector<std::string> messages;

		[[nodiscard]] int total() const;
		[[nodiscard]] int count(ValidationCheck check) const;
		[[nodiscard]] bool clean() const { return total() == 0; }
		void clear();
	};

	/// @brief Thrown by Population::reportViolation when the active
	/// ValidationPolicy is Throw.
	class ValidationError : public std::runtime_error
	{
	public:
		ValidationError(ValidationCheck check, const std::string& message);
		[[nodiscard]] ValidationCheck getCheck() const { return check; }
	private:
		ValidationCheck check;
	};

	/// @brief Configuration for a NEAT population run.
	struct PopulationParameters
	{
		int size; ///< Number of solutions per generation; must be greater than 0.
		int currentGeneration{0};
		int numGenerations;
		double targetFitness; ///< Evolution stops early when every partial fitness of the best solution exceeds this value.
		bool parallelEvolution; ///< Evaluate solutions concurrently via std::async.

		explicit PopulationParameters(int size = 100, int numGenerations = 1000, double targetFitness = 0.95, bool parallelEvolution = true);
	};

	/// @brief Runtime flags for pausing or stopping evolution from an external thread.
	struct PopulationControl
	{
		std::atomic<bool> pause;
		std::atomic<bool> stop;

		explicit PopulationControl(bool pause = false, bool stop = false);
	};

	struct PopulationStatistics
	{
		std::chrono::time_point<std::chrono::steady_clock> start;
		std::chrono::time_point<std::chrono::steady_clock> end;
		long long duration{};

		PopulationStatistics() = default;
	};

	/// @brief Per-generation snapshot of population health metrics.
	struct PerGenerationStatistics
	{
		double averageFitness = 0.0F;
		double bestFitness = 0.0F;
		int numberOfSpecies = 0;
		int numberOfActiveSpecies = 0;
		int innovationNumber = 0;
		double averageGenomeSize = 0.0;
		double averageConnectionGenes = 0.0;
		double averageFieldGenes = 0.0;

		PerGenerationStatistics() = default;
	};

	/// @brief Manages a NEAT population: speciation, evaluation, reproduction, and selection.
	///
	/// Call @c initialize() once, then @c evolve() to run the full evolutionary loop.
	/// Evolution stops when every partial fitness of the best solution exceeds @c PopulationParameters::targetFitness or
	/// @c numGenerations is exhausted. Use @c pause() / @c stop() for interactive control.
	class Population
	{
		friend class PopulationFileManager;
		// Test-only accessor for issue #59's regression tests -- see
		// tests/test_population_access.h for why direct access is needed.
		friend class test::PopulationTestAccess;
	private:
		PopulationParameters parameters;
		std::vector<SolutionPtr> solutions;
		std::vector<std::shared_ptr<Species>> speciesList;
		SolutionPtr bestSolution;
		std::vector<SolutionPtr> champions;
		PopulationControl control;
		PopulationStatistics statistics;
		PerGenerationStatistics perGenStatistics;
		bool hasFitnessImproved{};
		int generationsWithoutImprovement = 0;
		double previousBestFitness = 0.0;
		SolutionPtr previousBestSolution;
		std::vector<double> bestFitnessHistory;
		std::vector<int> bestSolutionIdHistory;
		std::vector<Genome> bestSolutionGenomeHistory;
		std::unique_ptr<PopulationFileManager> fileManager;
		ValidationReport validationReport;
		ValidationPolicy validationPolicy = defaultValidationPolicy;
		ParetoArchive paretoArchive;
		/// Ids of the front-0 solutions the archive accepted this generation.
		std::vector<int> acceptedIntoArchive;
		/// Each living species' own non-dominated history, keyed by species id: what a
		/// member has to beat for its species to count as improving in Pareto mode.
		std::map<int, ParetoArchive> speciesFronts;

		/// @brief What the latest Pareto ranking found beyond the archive's accepted ids:
		/// the inputs of the archive-empty improvement fallback and of the per-generation
		/// DEBUG sentence.
		struct ParetoRankingSummary
		{
			size_t numberOfFronts{0};
			size_t frontZeroSize{0};
			size_t frontZeroFeasible{0};
			/// The smallest constraint violation in the population.
			double lowestViolation{0.0};
			/// The archive is empty and lowestViolation fell below every violation offered
			/// to it before by more than SelectionConstants::archiveEpsilon.
			bool lowestViolationImproved{false};
		};
		ParetoRankingSummary rankingSummary;

		// Not thread-safe; only ever called from upkeep()/speciate(), both
		// main-thread. Must not be called from the parallel evaluate() path.
		void reportViolation(ValidationCheck check, const std::string& message);
	public:
		Population(const PopulationParameters& parameters,
			const SolutionPtr& initialSolution,
			bool enableFileIO = true);
		~Population();
		Population(const Population& other) = delete;
		Population(Population&& other) = delete;
		Population& operator=(const Population& other) = delete;
		Population& operator=(Population&& other) = delete;

		/// @brief Resets the process-global Species/Genome/Solution id and innovation
		/// counters to zero.
		/// @details These counters are static state shared by every Species, Genome,
		/// and Solution in the process, not per-Population data -- resetting them
		/// while another Population (or any Solution/Genome/Species it owns) is still
		/// alive will renumber new instances into ids already in use by that survivor.
		/// Call this only when no such survivor exists, e.g. between fully independent
		/// runs in the same process, or from test setup. Never called automatically by
		/// Population itself (in particular, not from the destructor -- destroying one
		/// Population must not corrupt another's numbering).
		static void resetGlobalCounters();

		void initialize() const;
		void evolve();

		[[nodiscard]] SolutionPtr getBestSolution() const { return bestSolution; }
		std::vector<std::shared_ptr<Species>> getSpeciesList() { return speciesList; }
		[[nodiscard]] std::vector<SolutionPtr> getSolutions() const { return solutions; }
		[[nodiscard]] int getSize() const { return parameters.size; }
		[[nodiscard]] int getCurrentGeneration() const { return parameters.currentGeneration; }
		[[nodiscard]] int getNumGenerations() const { return parameters.numGenerations; }
		[[nodiscard]] bool isInitialized() const { return !solutions.empty(); }
		[[nodiscard]] const std::vector<double>& getBestFitnessHistory() const { return bestFitnessHistory; }
		[[nodiscard]] const std::vector<int>& getBestSolutionIdHistory() const { return bestSolutionIdHistory; }
		[[nodiscard]] const std::vector<Genome>& getBestSolutionGenomeHistory() const { return bestSolutionGenomeHistory; }

		[[nodiscard]] const ValidationReport& getValidationReport() const { return validationReport; }
		void setValidationPolicy(ValidationPolicy policy) { validationPolicy = policy; }

		/// Sets the ValidationPolicy every subsequently constructed Population
		/// starts with. Production code never calls this (default stays Log);
		/// the test binary calls it once at startup (see tests/entry.cpp) so
		/// invariant violations throw during tests without any production
		/// behaviour change.
		static void setDefaultValidationPolicy(ValidationPolicy policy) { defaultValidationPolicy = policy; }

		void setSize(const int size) { parameters.size = size; }
		void setNumGenerations(const int numGenerations) { parameters.numGenerations = numGenerations; }

		void pause() { control.pause = true; }
		void resume() { control.pause = false; }
		void stop() { control.stop = true; }
		void start() { control.stop = false; }
	private:
		static inline ValidationPolicy defaultValidationPolicy = ValidationPolicy::Log;
		void evaluate() const;
		/// @brief Whether this generation's solutions are Pareto-ranked: always in Pareto
		/// mode, where selection reads the ranks, and in scalar mode only when
		/// objectives.jsonl is written.
		/// @return True in Pareto mode, or when file I/O is enabled and
		/// PopulationConstants::saveObjectives is set.
		[[nodiscard]] bool isRankingObjectives() const;
		/// @brief Non-dominated sort and crowding distance over the whole population,
		/// written into each solution's parameters with its rank-derived selection fitness,
		/// then offers front 0 to the archive and fills rankingSummary.
		/// Draws no random numbers. Runs on the main thread, after evaluate().
		void rankObjectives();
		/// @brief Offers each member of @p front to the Pareto archive and records the ids it
		/// accepts this generation in acceptedIntoArchive.
		/// @param points The ranked population, indexed like solutions.
		/// @param front Indices of front 0.
		void offerFrontToArchive(std::span<const RankedPoint> points, std::span<const size_t> front);
		/// @brief Fills rankingSummary from this generation's ranking.
		/// @param points The ranked population, indexed like solutions.
		/// @param fronts The fronts nonDominatedSort() returned for @p points.
		/// @param violationToBeat The archive's bestViolationSeen() before this generation's offers.
		void summarizeRanking(std::span<const RankedPoint> points,
			const std::vector<std::vector<size_t>>& fronts, double violationToBeat);
		/// @brief The Pareto-mode "population improved" signal of the latest ranking.
		/// @return True if the archive accepted a point, or, while the archive is empty,
		/// the lowest violation fell by more than SelectionConstants::archiveEpsilon.
		[[nodiscard]] bool hasParetoFrontImproved() const;
		/// @brief The Pareto-mode "species improved" signal: offers @p species' ranked
		/// members to the species' own front (its non-dominated history).
		/// @details The species' counterpart of scalar mode's "the champion beat the
		/// species' best fitness". It reads @p species' current members, so it must run
		/// after speciate() has placed them.
		/// @param species The species whose members are offered.
		/// @return True if its front accepted a member, or, while that front is empty, its
		/// lowest violation fell by more than SelectionConstants::archiveEpsilon.
		[[nodiscard]] bool offerSpeciesToItsFront(const Species& species);
		/// @brief Drops the fronts of species that are no longer in speciesList.
		void forgetFrontsOfExtinctSpecies();
		/// @brief Logs the per-generation Pareto DEBUG sentence: fronts, front 0, archive,
		/// and how many species improved and how many are stagnant.
		/// @param improvedSpecies Number of species that improved this generation.
		void logParetoProgress(int improvedSpecies) const;
		void speciate();
		/// @brief Picks every species' champion. A species improves when its champion's
		/// fitness rises: in scalar mode, and in Pareto mode with StagnationSignal::Fitness.
		/// In Pareto mode with StagnationSignal::Front it improves when
		/// offerSpeciesToItsFront() says so. Pareto mode then logs its per-generation
		/// DEBUG sentence.
		void assignChampions();
		void reproduceAndSelect();

		[[nodiscard]] bool endConditionMet() const;

		void startup();
		void upkeep();
		void cleanup();
		void createInitialSolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;

		void assignToSpecies(const SolutionPtr& solution);
		std::shared_ptr<Species> findSpecies(const SolutionPtr& solution);
		/// @brief The active species whose champion is preferred (Solution::isPreferredTo())
		/// over every other champion. In scalar mode a champion must also have a positive fitness.
		/// @return That species, or nullptr if there is none.
		[[nodiscard]] std::shared_ptr<Species> getBestActiveSpecies() const;

		void calculateAdjustedFitness();
		void assignOffspringToSpecies();
		void clearSpeciesOffspring() const;
		bool hasFitnessImprovedOverTheLastGenerations();
		void assignOffspringToTopTwoSpecies();
		void sortSpeciesListByChampionFitness();
		void assignOffspringBasedOnAdjustedFitness() const;
		void reassignOffspringIfFitnessIsStagnant() const;

		void pruneWorsePreformingSolutions() const;
		void replaceEntirePopulationWithOffspring();
		void preserveGlobalBestSolution();
		void mutate();

		void upkeepBestSolution();
		void upkeepChampions();
		void upkeepPerGenerationStatistics();
		void updateGenerationAndAges();
		void validateElitism();
		void validateUniqueSolutions();
		void validatePopulationSize();
		void validateUniqueGenesInGenomes();
		void validateUniqueKernelAndNeuralFieldPtrs();
		void validateIfSpeciesHaveUniqueRepresentative();
		void validateAssignmentIntoSpecies();

		void print() const;

		static void resetGenerationalInnovations();
		void clearLastMutations() const;

		void logSolutions() const;
		void logSpecies() const;
		void logOverview() const;
	};
}