#include <bits/stdc++.h>

const size_t POP_SIZE = 100;
const size_t GENERATIONS = 100000;
const size_t MAX_STAGNATION = 250;
const size_t MAX_STAGNATION_RESET = 5;
double MUTATION_RATE = 0.1;
const double TIME_LIMIT = 1.8; // seconds

std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::uniform_int_distribution<int> binary_dist(0, 1);

struct Item {
	long long value;
	long long weight;
	double ratio() const {
		if (weight == 0) return std::numeric_limits<double>::infinity();
		return (double)value / weight;
	}
};

using Individual = std::vector<bool>;
std::vector<Item> items;

size_t number_of_items;
long long max_weight;

std::pair<long long, long long> compute_value_weight(const Individual &ind) {
	long long total_value = 0;
	long long total_weight = 0;
	for (size_t i = 0; i < ind.size(); ++i) {
		if (ind[i]) {
			total_value += items[i].value;
			total_weight += items[i].weight;
		}
	}
	return {total_value, total_weight};
}

Individual greedy_by_ratio(const std::vector<Item> &items, long long max_weight) {
	size_t n = items.size();
	std::vector<size_t> indices(n);
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
		return items[a].ratio() > items[b].ratio();
	});
	Individual ind(n, false);
	long long current_weight = 0;
	for (size_t idx : indices) {
		if (current_weight + items[idx].weight <= max_weight) {
			ind[idx] = true;
			current_weight += items[idx].weight;
		}
	}
	return ind;
}

Individual greedy_by_value(const std::vector<Item> &items, long long max_weight) {
	size_t n = items.size();
	std::vector<size_t> indices(n);
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
		return items[a].value > items[b].value;
	});
	Individual ind(n, false);
	long long current_weight = 0;
	for (size_t idx : indices) {
		if (current_weight + items[idx].weight <= max_weight) {
			ind[idx] = true;
			current_weight += items[idx].weight;
		}
	}
	return ind;
}

void local_improve(Individual &ind) {
    auto [val, wt] = compute_value_weight(ind);

    for (size_t i = 0; i < ind.size(); ++i) {
        if (!ind[i] && wt + items[i].weight <= max_weight) {
            ind[i] = true;
            wt += items[i].weight;
        }
    }
}

Individual repair(const Individual &ind) {
    Individual repaired = ind;
    auto [_, total_weight] = compute_value_weight(repaired);

    if (total_weight <= max_weight) return repaired;

    std::vector<size_t> selected;
    for (size_t i = 0; i < repaired.size(); ++i)
        if (repaired[i])
            selected.push_back(i);

    if (selected.empty()) return repaired; // safety

    // sort by increasing ratio (worst first)
    std::sort(selected.begin(), selected.end(),
        [&](size_t a, size_t b) {
            return items[a].ratio() < items[b].ratio();
        });

    // shuffle bottom-k to keep randomness
    const size_t K = std::min<size_t>(5, selected.size());
    std::shuffle(selected.begin(), selected.begin() + K, rng);

    for (size_t idx : selected) {
        if (total_weight <= max_weight) break;
        repaired[idx] = false;
        total_weight -= items[idx].weight;
    }

    return repaired;
}

std::vector<Individual> initial_population(size_t item_count, const std::vector<Item> &items) {
	std::vector<Individual> population(POP_SIZE, Individual(item_count, false));
	
	// First individual uses greedy heuristic
	population[0] = greedy_by_ratio(items, max_weight);
	population[1] = greedy_by_value(items, max_weight);

	// Rest are random
	for (size_t i = 2; i < POP_SIZE; ++i) {
		for (size_t j = 0; j < item_count; ++j) {
			population[i][j] = binary_dist(rng);
		}
		// Repair if infeasible
		population[i] = repair(population[i]);
		// Local improvement
		local_improve(population[i]);
	}
	return population;
}

long long evaluate(const Individual &ind) {
	auto [total_value, total_weight] = compute_value_weight(ind);
	return (total_weight <= max_weight) ? total_value : 0;
}

void repopulate(std::vector<Individual> &population, const Individual &best_individual) {
	std::vector<std::pair<long long, Individual>> ranked;
	ranked.reserve(population.size());

	for (const auto &ind : population)
		ranked.emplace_back(evaluate(ind), ind);

	std::sort(ranked.begin(), ranked.end(),
			[](auto &a, auto &b) { return a.first > b.first; });

	const size_t elite_count = POP_SIZE / 10;      // 10%
	const size_t reset_count = POP_SIZE * 3 / 10;  // 30%

	std::vector<Individual> new_population;

	// 1) Keep elites
	for (size_t i = 0; i < elite_count; ++i)
		new_population.push_back(ranked[i].second);

	// 2) Add greedy seeds (important)
	new_population.push_back(greedy_by_ratio(items, max_weight));
	new_population.push_back(greedy_by_value(items, max_weight));

	// 3) Inject randomized individuals
	while (new_population.size() < elite_count + reset_count) {
		Individual ind(number_of_items, false);
		for (size_t j = 0; j < number_of_items; ++j)
			ind[j] = binary_dist(rng);
		new_population.push_back(repair(ind));
	}

	// 4) Fill the rest from previous population
	size_t idx = elite_count;
	while (new_population.size() < POP_SIZE && idx < ranked.size()) {
		new_population.push_back(ranked[idx++].second);
	}

	population = std::move(new_population);
}


Individual tournament_selection(const Individual &ind1, const Individual &ind2, const Individual &ind3) {
	long long val1 = evaluate(ind1);
	long long val2 = evaluate(ind2);
	long long val3 = evaluate(ind3);
	if (val1 >= val2 && val1 >= val3) return ind1;
	if (val2 >= val1 && val2 >= val3) return ind2;
	return ind3;
}

std::vector<Individual> select_parents(const std::vector<Individual> &population) {
	std::vector<Individual> parents;
	std::uniform_int_distribution<size_t> pop_dist(0, POP_SIZE - 1);
	for (size_t i = 0; i < POP_SIZE; ++i) {
		size_t idx1 = pop_dist(rng);
		size_t idx2 = pop_dist(rng);
		size_t idx3 = pop_dist(rng);
		// Select the best among three
		parents.push_back(tournament_selection(population[idx1], population[idx2], population[idx3]));
	}
	return parents;
}

Individual mutate(const Individual &ind) {
	Individual mutated = ind;
	for (size_t i = 0; i < ind.size(); ++i) {
		if (uniform_dist(rng) < MUTATION_RATE) {
			mutated[i] = !mutated[i];
		}
	}
	return repair(mutated);
}

std::pair<Individual, Individual> crossover(const Individual &parent1, const Individual &parent2) {
	std::uniform_int_distribution<size_t> point_dist(0, parent1.size() - 1);
	size_t point = point_dist(rng);
	Individual child1 = parent1;
	Individual child2 = parent2;
	for (size_t i = point; i < parent1.size(); ++i) {
		child1[i] = parent2[i];
		child2[i] = parent1[i];
	}
	return {repair(child1), repair(child2)};
}

std::vector<Individual> create_next_generation(const std::vector<Individual> &parents) {
	std::vector<Individual> next_generation;
	std::uniform_int_distribution<size_t> parent_dist(0, parents.size() - 1);
	for (size_t i = 0; i < POP_SIZE / 2; ++i) {
		size_t idx1 = parent_dist(rng);
		size_t idx2 = parent_dist(rng);
		auto [child1, child2] = crossover(parents[idx1], parents[idx2]);
		next_generation.push_back(mutate(child1));
		next_generation.push_back(mutate(child2));
	}
	return next_generation;
}

long long genetic_algorithm(const std::vector<Item> &items) {
	auto population = initial_population(number_of_items, items);

	// Initialize best from initial population
	long long best_value = evaluate(population[0]);
	Individual best_individual = population[0];

	size_t stagnation_counter = 0;
	size_t stagnation_resets = 0;
	clock_t start_time = clock();

	for (size_t generation = 0; generation < GENERATIONS; ++generation) {
		if ((double)(clock() - start_time) / CLOCKS_PER_SEC > TIME_LIMIT) {
			break;
		}

		// Find best in current population
		long long current_best = 0;
		size_t best_idx = 0;
		for (size_t i = 0; i < population.size(); ++i) {
			long long val = evaluate(population[i]);
			if (val > current_best) {
				current_best = val;
				best_idx = i;
			}
		}
		
		if (current_best > best_value) {
			best_value = current_best;
			best_individual = population[best_idx];
			stagnation_counter = 0;
		} else {
			stagnation_counter++;
			if (stagnation_counter >= MAX_STAGNATION) {
				stagnation_counter = 0;
				stagnation_resets++;
				#ifndef ONLINE_JUDGE
				std::cout << "Stagnation reset " << stagnation_resets << " at generation " << generation << ", best value: " << best_value << std::endl;
				#endif
				if (stagnation_resets >= MAX_STAGNATION_RESET) {
					break;
				}
				// Reinitialize 30% of the population
				repopulate(population, best_individual);
			}
		}
		
		auto parents = select_parents(population);
		population = create_next_generation(parents);
		
		// Elitism: preserve best individual
		population[0] = best_individual;
	}
	return best_value;
}


int main() {
#ifdef ONLINE_JUDGE
	std::ios::sync_with_stdio(false);
	std::cin.tie(nullptr); std::cout.tie(nullptr);
#endif

	std::cin >> number_of_items >> max_weight;
	MUTATION_RATE = 1.0 / number_of_items;
	items.resize(number_of_items);
	for (size_t i = 0; i < number_of_items; ++i) {
		std::cin >> items[i].weight >> items[i].value;
	}
	long long best_value = genetic_algorithm(items);
	std::cout << best_value << std::endl;
	return 0;
}