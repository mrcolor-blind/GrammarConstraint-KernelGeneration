volumes/

Abstracción del almacenamiento persistente Modal.

Tu proyecto NO debería saber rutas hardcodeadas tipo:

/data/results/

Debe pedir:

artifact_store.save(...)