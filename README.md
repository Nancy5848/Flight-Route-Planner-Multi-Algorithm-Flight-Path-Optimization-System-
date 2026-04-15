The Flight Route Planner is an interactive desktop application built with Python, Tkinter, and Matplotlib for visualizing aviation routes.
It implements seven graph algorithms: BFS, DFS, Dijkstra, A*, Floyd-Warshall, Prim, and Kruskal for route optimization.
The system features a database of 40+ international airports with real-world coordinates and 409 flight connections.
Great-circle distances between airports are calculated using the Haversine formula to determine accurate route weights.
Users can select origin and destination airports, choose any algorithm, and view the optimal path on an interactive world map.
The "Compare All" feature runs all algorithms simultaneously, displaying each route with a unique color-coded path.
Real-time execution time measurements allow users to compare algorithm performance side-by-side.
The bottom panel displays algorithm performance cards, a benchmark chart, and detailed route information.
Prim and Kruskal algorithms generate Minimum Spanning Trees connecting all 40 airports with minimum total distance.
This project serves as both an educational tool for learning graph algorithms and a practical aviation route planning demonstration.
