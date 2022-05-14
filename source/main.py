from functools import partial
from random import choices, randint, random, randrange
import time
from typing import List, Callable, Tuple
from collections import namedtuple

# Generic representation of a solution
Genome = List[int]
Population = List[Genome]
FitnessFunction = Callable[[Genome], int]
PopulateFunction = Callable[[], Population]
SelectionFunction = Callable[[Population, FitnessFunction], Tuple[Genome, Genome]]
CrossoverFunction = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunction = Callable[[Genome], Genome]
Object = namedtuple('Object', ['name', 'value', 'weight'])

# Examples
first_example = [
    Object('Laptop', 500, 2200),
    Object('Headphones', 150, 160),
    Object('Coffee Mug', 60, 350),
    Object('Notepad', 40, 333),
    Object('Water Bottle', 30, 192),
]

second_example = [
    Object('Mints', 5, 25),
    Object('Socks', 10, 38),
    Object('Tissues', 15, 80),
    Object('Phone', 500, 200),
    Object('Baseball Cap', 100, 70)
] + first_example


# Function to generate a random new solution
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


# Function to generate multiple solutions
def generate_population(size: int, length: int) -> Population:
    return [generate_genome(length) for _ in range(size)]


# Function to calculate the fitness of a solution
def fitness(genome: Genome, objects: List[Object], weight_limit: int) -> int:
    if len(genome) != len(objects):
        raise ValueError("genome and objects must be of same length")
    weight, value = 0, 0
    for i, obj in enumerate(objects):
        if genome[i] == 1:
            weight += obj.weight
            value += obj.value
            if weight > weight_limit:
                return 0
    return value


# Selection function for the solutions for the next generation
def selection_pair(population: Population, fitness_function: FitnessFunction) -> Population:
    return choices(
        population = population,
        weights = [fitness_function(genome) for genome in population],
        k = 2
    )


# crossover function for the solutions
def single_point_crossover(a: Genome, b: Genome) -> Genome:
    if len(a) != len(b):
        raise ValueError("a and b must be of same length")

    if len(a) < 2:
        return a, b

    crossover_point = randint(1, len(a) - 1)
    return a[0:crossover_point] + b[crossover_point:], b[0:crossover_point] + a[crossover_point:]


# mutation function for the solutions
def mutation(genome: Genome, times: int = 1, mutation_rate: float = 0.8) -> Genome:
    for _ in range(times):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > mutation_rate else abs(genome[index] - 1)
    return genome


# Funtion to run the genetic algorithm
def genetic_algorithm(
    poulate_function: PopulateFunction,
    fitness_function: FitnessFunction,
    fitness_limit: int,
    selection_function: SelectionFunction = selection_pair,
    crossover_function: CrossoverFunction = single_point_crossover,
    mutation_function: MutationFunction = mutation,
    generation_limit: int = 100) -> Tuple[Population, int]:

    population = poulate_function()
    print(f"Initial population: {population}")
    for i in range(generation_limit):
        print(f"Generation {i}")
        population = sorted(
            population,
            key=lambda genome: fitness_function(genome),
            reverse=True
        )
        print(f"fitness: {fitness_function(population[0])}")
        if fitness_function(population[0]) >= fitness_limit:
            break
        
        next_generation = population[0:2]
        print('next_generation', next_generation)
        for j in range(len(population) // 2 - 1):
            parentes = selection_function(population, fitness_function)
            a_offspring, b_offspring = crossover_function(parentes[0], parentes[1])
            offspring_a, offspring_b = mutation_function(a_offspring), mutation_function(b_offspring)
            next_generation.append(offspring_a)
            next_generation.append(offspring_b)

        population = next_generation
   
    population = sorted(
        population,
        key=lambda genome: fitness_function(genome),
        reverse=True
    )
    return population, i


def genome_to_objects(genome: Genome, objects: List[Object]) -> List[Object]:
    return [obj.name for i, obj in enumerate(objects) if genome[i] == 1]


if __name__ == "__main__":
    #objects = first_example
    objects = second_example
    start = time.time()
    population, generations = genetic_algorithm(
        poulate_function=partial(generate_population, size=10, length=len(objects)),
        fitness_function=partial(fitness, objects=objects, weight_limit=3000),
        #fitness_limit=740,
        fitness_limit=1310,
        generation_limit=100,
    )
    end = time.time()
    print(f"Generations: {generations}")
    print(f"Best solution: {genome_to_objects(population[0], objects)}")
    print(f"Best fitness: {fitness(population[0], objects, 10000)}")
    print(f"time: {end - start}")