# GDPR Spain Neural Search Engine

the search engine should understand intent, not keywords.

```bash

uv sync

uv run scripts/analyze.py data/gdpr-export-spain.json

uv run scripts/generate_embeddings.py data/gdpr-export-spain.json

uv run scripts/search_app.py 

```

dual-layered compliance challenge 
GDPR + LOPDGDD (specific national requirements)

The interpretation of both the GDPR and the LOPDGDD is continuously shaped by new guidance issued by the European Data Protection Board (EDPB), Agencia Española de Protección de Datos (AEPD), Spanish National Courts and the European Court of Justice (ECJ).

https://huggingface.co/IIC/MEL

sources:
    [BOE](https://www.boe.es/datosabiertos/api/api.php)
    [CENDOJ](https://www.poderjudicial.es/search/indexAN.jsp)
    [AEPD](https://www.aepd.es/es)
    [EDPB](https://edpb.europa.eu/edpb_en)

https://www.enforcementtracker.com/

[CORENN](https://blog.wilsonl.in/corenn/)
https://blog.wilsonl.in/search-engine/