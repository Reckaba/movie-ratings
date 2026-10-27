"""Execute part 2 of notebooks/02_recommender.ipynb (cells tagged 'part2'), keeping part 1 outputs."""
import copy

import nbformat
from nbclient import NotebookClient

PATH = "notebooks/02_recommender.ipynb"
nb = nbformat.read(PATH, as_version=4)
# Every code cell except part 1, which needs data/movies.db (not in the repository)
run_idx = [i for i, c in enumerate(nb.cells)
           if c.cell_type == "code" and "movies.db" not in c.source]

sub = copy.deepcopy(nb)
sub.cells = [copy.deepcopy(nb.cells[i]) for i in run_idx]
NotebookClient(sub, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": "notebooks"}}).execute()
for j, i in enumerate(run_idx):
    nb.cells[i].outputs = sub.cells[j].outputs

count = 0
for cell in nb.cells:
    cell.metadata.pop("tags", None)
    if cell.cell_type == "code":
        count += 1
        cell.execution_count = count
        for out in cell.outputs:
            if "execution_count" in out:
                out["execution_count"] = count
nbformat.write(nb, PATH)
print("executed cells", run_idx)
