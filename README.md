# BIPA — inferência e avaliação do checkpoint publicado

O projeto usa o [modelo BIPA já treinado](https://huggingface.co/thiagomonteles/BIPA_g2p_Multidialect_Byt5) e o [dataset BIPA](https://huggingface.co/datasets/thiagomonteles/BIPA). **Não treina o modelo.**

Na primeira execução, é necessário ter internet para baixar o modelo e o dataset. Ambos são fixados em revisões específicas no código para que as entradas da avaliação não mudem silenciosamente.

## Instalar

No terminal, na pasta do projeto:

```powershell
python -m pip install -r requirements.txt
```

## Gerar pronúncias

```powershell
python bipa.py --dialect "[brasil]" --text casa
```

Para várias palavras:

```powershell
python bipa.py --dialect "[são-paulo]" --text carro noite casa
```

Dialetos aceitos: `[brasil]`, `[rio-de-janeiro]`, `[são-paulo]`, `[regiao-sul]`, `[regiao-nordeste]` e `[regiao-centro-oeste]`.

## Avaliar com Minimum PER

Avaliação completa no conjunto de teste e em todos os dialetos:

```powershell
python bipa.py --evaluate
```

Para comparar mais diretamente com a Tabela 4 do artigo, que avalia o dialeto geral brasileiro no teste:

```powershell
python bipa.py --evaluate --dialect "[brasil]"
```

Para avaliar a validação, substitua o split:

```powershell
python bipa.py --evaluate --split validation
```

`--limit 20` executa uma avaliação pequena de verificação; não use esse resultado como métrica final. A avaliação completa pode demorar, especialmente em CPU. O programa usa GPU NVIDIA quando disponível e CPU como alternativa.

O script baixa o TSV público, remove duplicatas exatas de dialeto/palavra/pronúncia e cria partições agrupadas por dialeto e palavra, estratificadas por dialeto e comprimento da palavra. As proporções visam 75% treino, 15% validação e 10% teste, compatíveis com as quantidades reportadas no artigo. A semente fixa torna a partição reproduzível para as revisões fixadas; os índices originais do artigo não foram publicados, então os valores podem diferir.

Para cada par dialeto-palavra no split, o modelo gera uma pronúncia. O script compara a previsão com as referências daquele par e usa a referência de menor PER (distância de Levenshtein normalizada), como no artigo. O resultado inclui Minimum PER macro, micro e por dialeto. Para o dialeto `[brasil]` no teste, o artigo reporta 6,07%; não se espera necessariamente o mesmo valor devido à diferença possível nas partições e detalhes de segmentação.

As métricas e previsões são salvas em `runs/evaluation/`. O arquivo `metrics.json` registra o split, as revisões do modelo e dataset, a semente, a quantidade de pares e os resultados; `predictions.tsv` contém os detalhes por palavra.

## Arquivos e licenças

- `bipa.py`: inferência e avaliação do modelo publicado.
- `requirements.txt`: dependências do projeto.
- `2026.propor-1.47.pdf`: artigo do PROPOR 2026.
- `LICENSE`: licença do código deste repositório.

O dataset BIPA indica licença CC BY-SA 4.0; o modelo indica Apache-2.0. O PDF e esses materiais têm termos próprios. Consulte as páginas correspondentes antes de redistribuí-los e cite o artigo ao compartilhar resultados.
