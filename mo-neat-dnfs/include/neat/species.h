#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
    class Species
    {
    private:
		static int currentSpeciesId;
        int id = 0;
        int offspringCount{0};
        SolutionPtr representative;
        SolutionPtr champion;
        std::vector<SolutionPtr> members;
        std::vector<SolutionPtr> offspring;
        bool extinct{false};
        int age{0};
        bool hasFitnessImproved = true;
        int generationsSinceFitnessImproved = 0;

        /// Resets the stagnation counter on an improvement, and increments it otherwise.
        void recordImprovement(bool improved);
    public:
        Species();
		~Species() = default;
		Species(const Species& other) = default;
		Species(Species&& other) noexcept = default;
		Species& operator=(const Species& other) = default;
		Species& operator=(Species&& other) noexcept = default;
        void setRepresentative(const SolutionPtr& newRepresentative);
        void randomlyAssignRepresentative();
        /// @brief Sorts members by Solution::isPreferredTo(), makes the first the champion,
        /// and counts the generation as an improvement if it beats the previous champion's fitness.
        void assignChampion();
        /// @brief Sorts members by Solution::isPreferredTo() and makes the first the champion,
        /// recording an improvement decided by the caller (Pareto mode: the archive accepted
        /// one of this species' members this generation).
        /// @param improvedThisGeneration Whether this generation counts as an improvement.
        void assignChampion(bool improvedThisGeneration);

        [[nodiscard]] size_t size() const;
        void setOffspringCount(int count);
        [[nodiscard]] SolutionPtr getRepresentative() const;
        [[nodiscard]] SolutionPtr getChampion() const;
        [[nodiscard]] int getId() const;
        [[nodiscard]] double totalAdjustedFitness() const;
        [[nodiscard]] int getOffspringCount() const;
        [[nodiscard]] std::vector<SolutionPtr> getMembers() const;
        [[nodiscard]] bool isExtinct() const;
        [[nodiscard]] bool hasFitnessImprovedOverTheLastGenerations() const;
        void incrementAge();
        void extinguish()
        {
            extinct = true;
            representative = nullptr;
            champion = nullptr;
            members.clear();
            offspring.clear();
        }
        static void resetUniqueIdentifier()
        {
			currentSpeciesId = 0;
        }

        void addSolution(const SolutionPtr& solution);
        void removeSolution(const SolutionPtr& solution);
        /// @brief Returns true if @p solution's genome is within the compatibility distance threshold of this species' representative.
        [[nodiscard]] bool isCompatible(const SolutionPtr& solution) const;
        [[nodiscard]] bool contains(const SolutionPtr& solution) const;
        /// @brief Sorts members best first by Solution::isPreferredTo(): by fitness in scalar
        /// mode, by front and crowding distance in Pareto mode.
        void sortMembersByFitness();
        /// @brief Sorts members (see sortMembersByFitness()) and removes the floor(size() * ratio) worst-performing members.
        /// Reassigns the representative if it was among those removed.
        /// @param ratio Fraction of the current membership to remove, in [0, 1].
        void pruneWorsePerformingMembers(double ratio);
    	void crossover();
        void replaceMembersWithOffspring();
        void copyChampionToNextGeneration();

        [[nodiscard]] std::string toString() const;
        void print() const;
    };
}