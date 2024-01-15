import streamlit as st
from PIL import Image
from evol import Evolution, Population
from concurrent.futures import ThreadPoolExecutor
import random
import os
from copy import deepcopy

from painting import Painting


def score(x: Painting) -> float:
    current_score = x.image_diff(x.target_image)
    return current_score


def pick_best_and_random(pop, maximize=False):
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(
            evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness
        )
    else:
        mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


def pick_random(pop):
    mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


def mutate_painting(x: Painting, rate=0.04, swap=0.5, sigma=1) -> Painting:
    x.mutate_triangles(rate=rate, swap=swap, sigma=sigma)
    return deepcopy(x)


def mate(mom: Painting, dad: Painting):
    child_a, child_b = Painting.mate(mom, dad)
    return deepcopy(child_a)


def print_summary(
    pop, img_template="output%d.png", checkpoint_path="output"
) -> Population:
    avg_fitness = sum([i.fitness for i in pop.individuals]) / len(pop.individuals)

    st.write(
        "\nCurrent generation %d, best score %f, pop. avg. %f "
        % (pop.generation, pop.current_best.fitness, avg_fitness)
    )
    img = pop.current_best.chromosome.draw()
    img_placeholder = st.empty()
    img_placeholder.image(img, caption=f"Generation {pop.generation}")

    # display the best image after each generation
    st.image(pop.current_best.chromosome.draw(), caption=f"Generation {pop.generation}")

    if pop.generation % 50 == 0:
        pop.checkpoint(target=checkpoint_path, method="pickle")

    return pop


def main():
    st.title("Image Evolution App")
    st.write("Upload an image and see its evolution!")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file).convert("RGBA")

        st.sidebar.title("Evolution Parameters")
        num_triangles = st.sidebar.slider(
            "Number of Triangles", min_value=50, max_value=500, value=150, step=50
        )
        population_size = st.sidebar.slider(
            "Population Size", min_value=100, max_value=500, value=200, step=50
        )
        early_mutation_rate = st.sidebar.slider(
            "Early Mutation Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05
        )
        early_swap_rate = st.sidebar.slider(
            "Early Swap Rate", min_value=0.0, max_value=1.0, value=0.75, step=0.1
        )
        mid_mutation_rate = st.sidebar.slider(
            "Mid Mutation Rate", min_value=0.0, max_value=0.5, value=0.15, step=0.05
        )
        mid_swap_rate = st.sidebar.slider(
            "Mid Swap Rate", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
        late_mutation_rate = st.sidebar.slider(
            "Late Mutation Rate", min_value=0.0, max_value=0.5, value=0.05, step=0.05
        )
        late_swap_rate = st.sidebar.slider(
            "Late Swap Rate", min_value=0.0, max_value=1.0, value=0.25, step=0.1
        )

        pop = Population(
            chromosomes=[
                Painting(num_triangles, target_image, background_color=(255, 255, 255))
                for _ in range(population_size)
            ],
            eval_function=score,
            maximize=False,
        )

        early_evo = (
            Evolution()
            .survive(fraction=0.05)
            .breed(
                parent_picker=pick_random,
                combiner=mate,
                population_size=population_size,
            )
            .mutate(
                mutate_function=mutate_painting,
                rate=early_mutation_rate,
                swap=early_swap_rate,
            )
            .evaluate(lazy=False)
            .callback(print_summary)
        )

        mid_evo = (
            Evolution()
            .survive(fraction=0.15)
            .breed(
                parent_picker=pick_best_and_random,
                combiner=mate,
                population_size=population_size,
            )
            .mutate(
                mutate_function=mutate_painting,
                rate=mid_mutation_rate,
                swap=mid_swap_rate,
            )
            .evaluate(lazy=False)
            .callback(print_summary)
        )

        late_evo = (
            Evolution()
            .survive(fraction=0.05)
            .breed(
                parent_picker=pick_best_and_random,
                combiner=mate,
                population_size=population_size,
            )
            .mutate(
                mutate_function=mutate_painting,
                rate=late_mutation_rate,
                swap=late_swap_rate,
            )
            .evaluate(lazy=False)
            .callback(print_summary)
        )

        pop = pop.evolve(early_evo, n=200)
        pop = pop.evolve(mid_evo, n=300)
        pop = pop.evolve(late_evo, n=1000)


if __name__ == "__main__":
    main()
