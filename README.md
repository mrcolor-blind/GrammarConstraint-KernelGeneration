Grammar-Constrained GPU Kernel (Triton) Generation

Notebook Flujo TritonBench:
https://colab.research.google.com/drive/1r820uomX4ivjKng-Sw7Vncc6JWq9Rm6R?usp=sharing 

FLUJO FASE 1:

Operator Loader
        ↓
Prompt Builder
        ↓
LLM API Backend
        ↓
Code Extraction
        ↓
call@1
        ↓
exe@1
        ↓
speedup
        ↓
Experiment Save


FASE 2 y 3:
(Grammar constraint y IDE extension)

compiler/
constraints/
integrations/
storage/
IR/
semantic analysis/