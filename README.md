# Processamento de Imagens de Satélite e Análise Espacial

Este projeto contém um conjunto de funções em **Python** para processamento de imagens de satélite (ex.: Sentinel-2) e análise espacial com **GeoJSON**, **NDVI**, estatísticas zonais e geração de mapas.

## 🚀 Funcionalidades

- Leitura e validação de polígonos (`GeoJSON`).
- Explosão de geometrias multiparte em geometrias simples.
- Interseção de polígonos com Unidades de Conservação (UCs).
- Download, extração e organização de imagens de satélite.
- Recorte de imagens por polígonos (`rasterio.mask`).
- Cálculo de **NDVI** (Índice de Vegetação por Diferença Normalizada).
- Estatísticas zonais (média, mínimo, máximo, desvio padrão).
- Geração de mapas de NDVI em **PDF**.
- Análise temporal de séries de imagens NDVI.
- Implementação de índices de vegetação como **VARI**.

## 📂 Estrutura esperada de diretórios

```

.
├── GeoJson
│   ├── polygons\_test.GeoJSON
│   ├── layer\_UCs.GeoJSON
│
├── Imagem
│   ├── drive
│   │   ├── T22LHH\_20211006T132239\_4326.zip
│   │   ├── T22LHH\_20211220T132234.zip
│   │   ├── T22LHH\_20220228T132236.zip
│   │
│   ├── output
│   │   ├── NDVI.tif
│   │   ├── cliped\_by\_field.tif
│   │   └── Mapas/
│   │       ├── talhao\_0\_ndvi.pdf
│   │       ├── talhao\_1\_ndvi.pdf
│   │       └── ...
│   │
│   └── Teste.jpg
│
├── main.py
└── requirements.txt

````

## 📦 Dependências

O projeto utiliza as seguintes bibliotecas:

- `geopandas`
- `shapely`
- `rasterio`
- `rasterstats`
- `numpy`
- `matplotlib`
- `opencv-python`
- `zipfile`
- `shutil`
- `os`

### Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
````

Exemplo de `requirements.txt`:

```
geopandas
shapely
rasterio
rasterstats
numpy
matplotlib
opencv-python
```

## ▶️ Como executar

Os principais testes/funções estão organizados como exemplos de uso:

* `test_01` → Corrige geometrias inválidas.
* `test_02` → Calcula área dos polígonos em hectares.
* `test_03` → Calcula interseção com UCs.
* `test_04` → Gera NDVI a partir de imagem satélite.
* `test_05` → Calcula estatísticas zonais do NDVI.
* `test_06` → Gera mapas de NDVI em PDF.
* `test_07` → Alias para `test_05`.
* `test_08` → Série temporal de NDVI.
* `test_09` → Teste com índice **VARI** em imagens RGB.

### Exemplo de execução

```bash
python main.py
```

Ou, para rodar diretamente uma função no código:

```python
from main import test_04, test_06

# Gerar NDVI
ndvi_path = test_04()
print("NDVI salvo em:", ndvi_path)

# Criar mapas PDF
test_06()
```

## 🛰️ Dados de entrada

Os dados satelitais utilizados (ex.: Sentinel-2) devem ser baixados manualmente e colocados na pasta `Imagem/drive` conforme indicado em `check_files()`.
As funções esperam arquivos `.zip` com as bandas necessárias (`B04` e `B08`).

Links de exemplo já estão mapeados no código:

* [T22LHH\_20211006T132239\_4326](https://drive.google.com/file/d/1sV3TGy9OLnvhzAcZF5Qo3laSb2UXDJUN/view?usp=sharing)
* [T22LHH\_20211220T132234](https://drive.google.com/file/d/1vr_vnW-fZsoTyV5gkt7M7t73MlGzEcmC/view?usp=sharing)
* [T22LHH\_20220228T132236](https://drive.google.com/file/d/1y7ki4K21T2xUQx0d3JiKBTzOrZxFh8OG/view?usp=sharing)

## 📖 Referências

* Sentinel-2 User Guide
* [Shapely Documentation](https://shapely.readthedocs.io/)
* [Rasterio Documentation](https://rasterio.readthedocs.io/)
* [GeoPandas Documentation](https://geopandas.org/)

