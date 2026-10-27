import sys,json,sqlite3
import numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import embedding
embedding.DIRECTORY=root/'embedding-native-questions'
heldout='--holdout' in sys.argv
prompt_names=['evaluation-prompts-holdout.json' if heldout else 'evaluation-prompts.json']
vector_names=['evaluation-holdout-queries.npy' if heldout else 'evaluation-initial-queries.npy']
db=sqlite3.connect((embedding.DIRECTORY/'index.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)
if '--all-tests' in sys.argv:
    assert not heldout
    prompt_names.append('evaluation-prompts-holdout.json')
    vector_names.append('evaluation-holdout-queries.npy')
queries=[query for name in prompt_names for query in json.loads((embedding.DIRECTORY/name).read_text())['queries']]
# ponytail: replay cached query vectors when available; a fresh checkout encodes these same fixed prompts once.
vectors=(np.concatenate([np.load(embedding.DIRECTORY/name,allow_pickle=False) for name in vector_names])
         if all((embedding.DIRECTORY/name).exists() for name in vector_names) else None)
encoder=embedding.Encoder(1800*32*32) if vectors is None else None
names=[arg for arg in sys.argv[1:] if not arg.startswith('--')]
embedding.demo(db,encoder,queries,name=names[0] if names else 'evaluation-current',vectors=vectors)
db.close()
