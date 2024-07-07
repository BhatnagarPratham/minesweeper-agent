# minesweeper-like-agent

The aim of the project is to make a "Smart Agent" which will be able to locate the location of gold, while not encountering any mines.
For this project, inference logic is being used to eliminate the possibility of mine being in a cell and then finding a path to reach that cell. Thus the agent explores the map and uses inference logic to find the location of gold.

Glucose3 library is being used as a SAT solver, we add clauses depending on the number of mines nearby to a cell, and then ask the knowledge base if gold is present in any of the cells

Note: It is possible that it is impossible to find location of mine as no more cells can be explored due to them being mines

To run the Smart agent, run the following command in terminal:
```python Smart_Agent.py```