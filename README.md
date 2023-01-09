# Introduction-to-Ecology

Mimicry emerging from simple in vivo evolution

## Usage

```
usage: simulate.py [-h] [-K PREY [PREY ...]] [-T TOXICITY [TOXICITY ...]] [-H PREDATOR] [-G GENES] [-N GENERATIONS] [-P MUTATION_RATE] [-M MUTATION_SCALE] [-I INITIAL_SCALE] [-S SEED] [-O OUTPUT] [-C CMAP]

options:
  -h, --help            show this help message and exit
  -P PREY [PREY ...], --prey PREY [PREY ...]
                        Population size of prey species
  -T TOXICITY [TOXICITY ...], --toxicity TOXICITY [TOXICITY ...]
                        Toxicity of prey species
  -H PREDATOR, --predator PREDATOR
                        Population size of predator species
  -G GENES, --genes GENES
                        Length of genes
  -N GENERATIONS, --generations GENERATIONS
                        Number of generations
  -R MUTATION_RATE, --mutation_rate MUTATION_RATE
                        Probability of mutation
  -K MUTATION_SCALE, --mutation_scale MUTATION_SCALE
                        Scale of mutation
  -I INITIAL_SCALE, --initial_scale INITIAL_SCALE
                        Initial scale of deviation
  -S SEED, --seed SEED  Random seed
  -O OUTPUT, --output OUTPUT
                        Output directory
  -C CMAP, --cmap CMAP  color map to use
```

### Examples

- Two prey species with different toxicity (0.9 & 0.0) but same population size (500) and a predator species with population size 100.
  ```
  python3 simulate.py
  ```
- Increase the number of gene (default is 10) and run the simulation for more generations (default is 1000)
  ```
  python3 simulate.py -G 100 -N 10000
  ```
- Higher probability of mutation (default is 0.05), larger effect of mutation (default is 0.05) and larger deviation from the initial value (default is 0.1)
  ```
  python3 simulate.py -R 0.1 -K 0.1 -I 0.2
  ```
- Set the seed (default is 0), redirect output directory (default is fig) and select color map (default is rainbow)
  ```
  python3 simulate.py -S 2023 -O out -C viridis
  ```
- *If you have ffmpeg installed*, you can also use the post process script to compose figures into a video file
  ```
  python3 post-process.py -R 30 -D fig -O Population.mp4
  ```