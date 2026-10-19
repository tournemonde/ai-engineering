# Embedding sanity check (Session 7)

Model: `text-embedding-3-small` (1536 dimensions).  
Generated with `uv run python scripts/compare.py` from `estimator/`.

## Pair A — semantically close (expected high, ~> 0.6)

| Text A | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app |
| Text B | Authorization service using JSON Web Tokens for a banking application |
| **Cosine similarity** | **0.5957** |

## Pair B — unrelated (expected low, ~< 0.4)

| Text A | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app |
| Text B | Database migration from MySQL to PostgreSQL with zero downtime |
| **Cosine similarity** | **0.1920** |

## Pair C — generic / ambiguous (no fixed expectation)

| Text A | Backend services |
| Text B | API development |
| **Cosine similarity** | **0.5407** |

## Commentary

La pareja A queda justo por debajo del umbral orientativo 0.6, pero sigue siendo claramente más alta que la B (0.19): el modelo agrupa autenticación/OAuth/JWT frente a migración de bases de datos. La pareja C es la más interesante: dos frases muy cortas y genéricas obtienen ~0.54, casi en el rango de A, lo que muestra que sin contexto el embedding tiende a colapsar hacia un “espacio de desarrollo de software” compartido. Eso refuerza por qué en el chunker añadimos cabeceras con sector, proyecto y tecnología antes de embeber componentes reales.
