import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import random
import time

class PathPlannerGA:
    def __init__(self, start, goal, obstacles, population_size=50, generations=100, 
                 waypoints=5, mutation_rate=0.1, crossover_rate=0.9, elitism=0.1, 
                 blx_alpha=0.5, polynomial_eta=20, resolution_step=0.1, grid_width=None, grid_height=None):
        """
        Initialize the genetic algorithm path planner.
        
        Parameters:
        - start: (x, y) tuple for start position
        - goal: (x, y) tuple for goal position
        - obstacles: list of obstacles (each obstacle is (x, y, width, height) for rectangles)
        - population_size: number of individuals in population
        - generations: number of generations to run
        - waypoints: number of intermediate waypoints in each path
        - mutation_rate: probability of mutation
        - crossover_rate: probability of crossover
        - elitism: fraction of population to carry over unchanged
        - blx_alpha: parameter for BLX-alpha crossover
        - polynomial_eta: parameter for polynomial mutation
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.population_size = population_size
        self.generations = generations
        self.waypoints = waypoints
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.blx_alpha = blx_alpha
        self.polynomial_eta = polynomial_eta
        self.resolution_step = resolution_step
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Determine search space boundaries
        if self.grid_width is not None and self.grid_height is not None:
            self.min_x = 0
            self.max_x = self.grid_width
            self.min_y = 0
            self.max_y = self.grid_height
        else:
            all_points = [start, goal] + [(obs[0], obs[1]) for obs in obstacles]
            self.min_x = min(p[0] for p in all_points) - 2
            self.max_x = max(p[0] for p in all_points) + 2
            self.min_y = min(p[1] for p in all_points) - 2
            self.max_y = max(p[1] for p in all_points) + 2

        # Initialize population
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """Initialize population with random paths."""
        population = []
        for _ in range(self.population_size):
            # Generate random waypoints between start and goal
            waypoints = []
            for _ in range(self.waypoints):
                x = random.uniform(self.min_x, self.max_x)
                y = random.uniform(self.min_y, self.max_y)
                waypoints.append([x, y])
            
            # Create path: start + waypoints + goal
            path = [self.start] + waypoints + [self.goal]
            population.append(np.array(path))
        return population
    
    def calculate_path_length(self, path):
        """Calculate the total length of a path."""
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length
    
    def line_intersects_obstacle(self, p1, p2, obstacle):
        """Check if a line segment between p1 and p2 intersects with an obstacle."""
        # For rectangle obstacles (x, y, width, height)
        if len(obstacle) == 4:
            x, y, w, h = obstacle
            rect = Rectangle((x, y), w, h, fill=True)
            
            # Check if either point is inside the rectangle
            if (x <= p1[0] <= x + w and y <= p1[1] <= y + h) or \
               (x <= p2[0] <= x + w and y <= p2[1] <= y + h):
                return True
                
            # Check for line segment intersection with rectangle edges
            rect_edges = [
                [(x, y), (x + w, y)],     # top
                [(x + w, y), (x + w, y + h)], # right
                [(x + w, y + h), (x, y + h)], # bottom
                [(x, y + h), (x, y)]      # left
            ]
            
            for edge in rect_edges:
                if self.lines_intersect(p1, p2, edge[0], edge[1]):
                    return True
                    
        # For circular obstacles (x, y, radius)
        elif len(obstacle) == 3:
            x, y, r = obstacle
            # Check if either point is inside the circle
            if (np.linalg.norm(p1 - np.array([x, y]))) <= r or \
               (np.linalg.norm(p2 - np.array([x, y]))) <= r:
                return True
                
            # Check for line segment intersection with circle
            if self.line_circle_intersection(p1, p2, (x, y), r):
                return True
                
        return False
    
    def lines_intersect(self, a1, a2, b1, b2):
        """Check if two line segments a1-a2 and b1-b2 intersect."""
        # Implementation of line segment intersection test
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    def line_circle_intersection(self, p1, p2, center, radius):
        """Check if line segment p1-p2 intersects with circle."""
        # Vector d
        d = p2 - p1
        # Vector f
        f = p1 - center
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return False
            
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
            
        return False
    
    def calculate_collisions(self, path):
        """Calculate the number of collisions with obstacles."""
        collisions = 0
        for i in range(len(path) - 1):
            for obstacle in self.obstacles:
                if self.line_intersects_obstacle(path[i], path[i+1], obstacle):
                    collisions += 1
                    break  # count only one collision per segment
        return collisions
    
    def evaluate_path(self, path):
        """Evaluate a path based on length and collisions."""
        length = self.calculate_path_length(path)
        collisions = self.calculate_collisions(path)
        return length, collisions
    
    def rank_partitioning(self, population, resolution_step):
        """Implements nested partitioning with three objectives, aligning with partition-based ranking."""
        # Evaluate all individuals
        evaluated = []
        for idx, individual in enumerate(population):
            length, collisions = self.evaluate_path(individual)
            evaluated.append({
                'index': idx,
                'collisions': collisions,                        # Tier 1 (discrete)
                'length_bucket': math.floor(length / resolution_step), # Tier 2 (discrete)
                'length_raw': length                             # Tier 3 (tie-breaker)
            })

        # Recursive partitioning function
        def partition(group, priority_level):
            if priority_level > 3 or len(group) <= 1:
                return group
                
            if priority_level == 1:
                key = lambda x: x['collisions']
            elif priority_level == 2:
                key = lambda x: x['length_bucket']
            elif priority_level == 3:
                key = lambda x: x['length_raw']
            
            group.sort(key=key)
            partitioned = []
            current_val = None
            subgroup = []
            
            for ind in group:
                if key(ind) == current_val:
                    subgroup.append(ind)
                else:
                    if subgroup:
                        partitioned.extend(partition(subgroup, priority_level + 1))
                    subgroup = [ind]
                    current_val = key(ind)
            
            if subgroup:
                partitioned.extend(partition(subgroup, priority_level + 1))
                
            return partitioned

        # Apply nested partitioning
        ranked = partition(evaluated, 1)
        
        # Assign ranks based on partition order
        for rank, ind in enumerate(ranked, start=1):
            ind['rank'] = rank
        
        return [(x['index'], x['collisions'], x['length_bucket'], x['length_raw'], x['rank'])
                for x in ranked]    

        # Tournament selection
    def tournament_selection(self, ranked_population, tournament_size=3):
        selected = []
        for _ in range(self.population_size):
            competitors = random.sample(ranked_population, tournament_size)
            winner = min(competitors, key=lambda x: x[4])  # Select by rank (5th item)
            selected.append(self.population[winner[0]]) # Append the actual path    
        return selected

    
    def blx_alpha_crossover(self, parent1, parent2):
        """BLX-alpha crossover between two parents."""
        child1 = []
        child2 = []
        
        for wp1, wp2 in zip(parent1, parent2):
            # For each waypoint coordinate
            x_min = min(wp1[0], wp2[0])
            x_max = max(wp1[0], wp2[0])
            x_range = x_max - x_min
            
            y_min = min(wp1[1], wp2[1])
            y_max = max(wp1[1], wp2[1])
            y_range = y_max - y_min
            
            # Create new waypoints within extended range
            new_x1 = random.uniform(x_min - self.blx_alpha * x_range, x_max + self.blx_alpha * x_range)
            new_y1 = random.uniform(y_min - self.blx_alpha * y_range, y_max + self.blx_alpha * y_range)
            
            new_x2 = random.uniform(x_min - self.blx_alpha * x_range, x_max + self.blx_alpha * x_range)
            new_y2 = random.uniform(y_min - self.blx_alpha * y_range, y_max + self.blx_alpha * y_range)
            
            # Clip to search space boundaries
            new_x1 = np.clip(new_x1, self.min_x, self.max_x)
            new_y1 = np.clip(new_y1, self.min_y, self.max_y)
            new_x2 = np.clip(new_x2, self.min_x, self.max_x)
            new_y2 = np.clip(new_y2, self.min_y, self.max_y)
            
            child1.append([new_x1, new_y1])
            child2.append([new_x2, new_y2])
        
        return np.array(child1), np.array(child2)
    
    def polynomial_mutation(self, individual):
        """Apply polynomial mutation to an individual."""
        mutated = individual.copy()
        
        for i in range(1, len(mutated)-1):  # Don't mutate start and goal
            if random.random() < self.mutation_rate:
                # Mutate x coordinate
                u = random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (self.polynomial_eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.polynomial_eta + 1))
                
                mutated[i][0] += delta * (self.max_x - self.min_x)
                mutated[i][0] = np.clip(mutated[i][0], self.min_x, self.max_x)
                
                # Mutate y coordinate
                u = random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (self.polynomial_eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.polynomial_eta + 1))
                
                mutated[i][1] += delta * (self.max_y - self.min_y)
                mutated[i][1] = np.clip(mutated[i][1], self.min_y, self.max_y)
        
        return mutated
    
    def evolve(self):
        """Run the genetic algorithm for path planning."""
        best_path = None
        best_length = float('inf')
        history = []
        last_generation_rankings = None
        
        for generation in range(self.generations):
            # Evaluate population
            evaluations = [self.evaluate_path(ind) for ind in self.population]

            # Create a list of (index, length, collisions) tuples
            indexed_evals = [(i, e[0], e[1]) for i, e in enumerate(evaluations)]
        
            # Sort by composite score (heavily penalize collisions)
            sorted_evals = sorted(indexed_evals, key=lambda x: (x[2], x[1]))

            # Store the rankings for the last generation
            if generation == self.generations - 1:
                last_generation_rankings = sorted_evals
            
            # Find best individual
            current_best_index = sorted_evals[0][0]
            # current_best_index = np.argmin([e[0] + 10000 * e[1] for e in evaluations])  # heavily penalize collisions
            current_best = self.population[current_best_index]
            current_length, current_collisions = evaluations[current_best_index]
            
            # Update global best (only if no collisions)
            if current_collisions == 0 and current_length < best_length:
                best_path = current_best.copy()
                best_length = current_length
            
            # Store generation statistics
            history.append({
                'generation': generation,
                'best_length': current_length,
                'best_collisions': current_collisions,
                'avg_length': np.mean([e[0] for e in evaluations]),
                'avg_collisions': np.mean([e[1] for e in evaluations]),
                'feasible_count': sum(1 for e in evaluations if e[1] == 0)
            })
            
            print(f"Generation {generation}: Best length = {current_length:.2f}, Collisions = {current_collisions}, Feasible = {sum(1 for e in evaluations if e[1] == 0)}/{self.population_size}")

            
            rankings = self.rank_partitioning(self.population, self.resolution_step)
            

            # Selection
            selected = self.tournament_selection(rankings, tournament_size=3)
            
            # Elitism: keep the best individuals
            elite_size = int(self.elitism * self.population_size)
            elite = [self.population[idx] for idx, _, _, _, _ in rankings[:elite_size]]
            
            # Crossover
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                parent1, parent2 = selected[i], selected[i+1]
                if random.random() < self.crossover_rate:
                    child1, child2 = self.blx_alpha_crossover(parent1, parent2)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1.copy(), parent2.copy()])
            
            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_rate:
                    offspring[i] = self.polynomial_mutation(offspring[i])
            
            # Create new population: elite + offspring
            self.population = elite + offspring[:self.population_size - elite_size]

        # Print final generation rankings
        if last_generation_rankings is not None:
            print("\nFinal Generation Rankings (Rank: Length, Collisions):")
            for idx, collisions, length_bucket, length, rank in rankings[:10]:  # top 10
                print(f"Rank {rank}: Length = {length:.2f}, Collisions = {collisions}")
                if rank % 10 == 0:  # Add a newline every 10 solutions for readability
                    print()
        
        # Return the best path found (with no collisions)
        return best_path, history
    
    def plot_path(self, path, title="Path"):
        """Plot the environment and path."""
        plt.figure(figsize=(10, 8))
        
        # Plot obstacles
        for obstacle in self.obstacles:
            if len(obstacle) == 4:  # U letter
                obs1 = Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], 
                                color='gray', alpha=0.5)
                plt.gca().add_patch(obs1)
            elif len(obstacle) == 3:  # T letter
                obs2 = Rectangle((obstacle[0], obstacle[1]), obstacle[2], 
                               color='gray', alpha=0.5)
                plt.gca().add_patch(obs2)
        
        # Plot start and goal
        plt.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        plt.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        # Plot path if it exists
        if path is not None:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
            plt.plot(path_x, path_y, 'bo', markersize=4)
        
        plt.xlim(self.min_x, self.max_x)
        plt.ylim(self.min_y, self.max_y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_statistics(self, history):
        """Plot the evolution statistics."""
        generations = [h['generation'] for h in history]
        best_lengths = [h['best_length'] for h in history]
        best_collisions = [h['best_collisions'] for h in history]
        avg_lengths = [h['avg_length'] for h in history]
        avg_collisions = [h['avg_collisions'] for h in history]
        feasible_counts = [h['feasible_count'] for h in history]
        
        plt.figure(figsize=(12, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(generations, best_lengths, 'b-', label='Best Length')
        plt.plot(generations, avg_lengths, 'r-', label='Average Length')
        plt.xlabel('Generation')
        plt.ylabel('Path Length')
        plt.title('Path Length Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(generations, best_collisions, 'b-', label='Best Collisions')
        plt.plot(generations, avg_collisions, 'r-', label='Average Collisions')
        plt.xlabel('Generation')
        plt.ylabel('Collision Count')
        plt.title('Collision Count Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(generations, feasible_counts, 'g-', label='Feasible Paths')
        plt.xlabel('Generation')
        plt.ylabel('Number of Feasible Paths')
        plt.title('Feasible Paths Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Define start, goal and obstacles
    start = (0, 0)
    goal = (20, 20)
    
    # Define obstacles as (x, y, width, height) for rectangles or (x, y, radius) for circles
    obstacles = [
        # (5.5, 4, 6, 1),        # Bottom horizontal of U
        # (5.5, 4, 1, 6),        # Left vertical of U
        # (11.5, 4, 1, 6),        # Right vertical of U

        (2, 2, 4, 1),  #  % [x, y, width, height]
        (3, 6, 2, 3),
        (7, 3, 1, 5),
        (6, 7, 3, 1)


        # (15, 5, 4, 4),    # Square 1 at (2,2)
        # # (7, 8, 4, 4),    # Square 2 at (6,2)
        # (0, 6, 4, 6),    # Rectangle 1 at (2,6)
        # (15, 14, 3, 5),     # Rectangle 2 at (6,6)


        # (50, 70, 4, 4),    # Square 1 at (2,2)
        # (30, 50, 4, 4),    # Square 2 at (6,2)
        # # (40, 40, 7, 7),    # Rectangle 1 at (2,6)
        # # (15, 14, 3, 3)     # Rectangle 2 at (6,6)
    ]
    
    # Create and run the GA path planner
    ga = PathPlannerGA(
        start=start,
        goal=goal,
        obstacles=obstacles,
        population_size=50,
        generations=50,
        waypoints=4,
        mutation_rate=0.1,
        crossover_rate=0.9,
        elitism=0.1,
        resolution_step=0.1,
        grid_width=20,
        grid_height=20,
    )
    
    # Plot initial random paths
    # ga.plot_path(ga.population[0], "Initial Random Path")
    
    # Run the GA
    # best_path, history = ga.evolve()

    # Measure computation time
    start_time = time.time()
    best_path, history = ga.evolve()
    end_time = time.time()

    # Calculate elapsed time
    computation_time = end_time - start_time    


    # Plot results
    if best_path is not None:
        print(f"Found feasible path with length: {ga.calculate_path_length(best_path):.2f}")
        print(f"Total collisions: {ga.calculate_collisions(best_path)}")
        print(f"Computation time: {computation_time:.4f} seconds")
        ga.plot_path(best_path, "Best Found Path")
    else:
        print("No feasible path found!")
        print(f"Total collisions: {ga.calculate_collisions(best_path)}")
        print(f"Computation time: {computation_time:.4f} seconds")
    
    # Plot statistics
    # ga.plot_statistics(history)