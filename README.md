# Análise de sentimento

## Tratamentos

### Stopwords

> *Stopwords são palavras comuns que normalmente não contribuem para o significado de uma frase*
> 

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/ce860fe5-b7a1-4362-ac53-59e9bbb78817/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220624%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220624T134547Z&X-Amz-Expires=86400&X-Amz-Signature=76edc073bc245ef1f680ad55d5e5165c313f8ef1143c84d67e4dba4dee82d2d1&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

### Steamming

> Técnica de remover sufixos e prefixos de uma palavra
> 

> Maior performance
> 

### Números

> Before: Comprei o produto dia 25 de fevereiro e hoje dia 29 de marco não fora entregue na minha residência. Não sei se os correios desse Brasil e péssimo ou foi a própria loja que demorou postar.
> 

> After: Comprei o produto dia numero de fevereiro e hoje dia numero de marco não fora entregue na minha residência. Não sei se os correios desse Brasil e péssimo ou foi a própria loja que demorou postar.
> 

### Negação

> Before: O material é bom, o problema é que a bolsa não fecha, não possui zíper, é como uma sacola. Isso me deixou insatisfeita, pois na foto não dá pra perceber e não há informação ou foto interna sobre isso.
> 

> After: O material é bom, o problema é que a bolsa negação fecha, negação possui zíper, é como uma sacola. Isso me deixou insatisfeita, pois na foto negação dá pra perceber e negação há informação ou foto interna sobre isso.
> 

## Treinamento

Para treinar um modelo de análise sentimental, precisamos do rótulo para aplicar em uma abordagem de Machine Learning supervisionada. > 1 se for positivo

> 0 se for negativo
> 

![https://i.imgur.com/6yb2MnV.png](https://i.imgur.com/6yb2MnV.png)

### Bag of Words

```json
{
	texto1 = "Os cursos de NLP da Alura utilizam Bag of Words"
	texto2 = "Aprendi Bag of Words perguntando no fórum da Alura"
}
```

[Untitled](https://www.notion.so/c21a351a2867462ea0868f21cd54ae8b)

### Cálculo

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9e8ee490-7c60-447a-9be1-160755c6a9f2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220624%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220624T134618Z&X-Amz-Expires=86400&X-Amz-Signature=f98dc836fb6f89f4bdb57447ee850e8cd6eea2ef7fb181fd559814287656051f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

Depois disso entra o algoritmo `Naive Bayes`, internamente ele utilizará esta informação para criar uma classificação para qualquer comentário recebido.

Imagine que ele receba o comentário: `"amei esse produto"`, para computar a pontuação positiva ele irá multiplicar individualmente a pontuação de cada palavra (por isso `Naive`) e do total presente nas fontes de dados.

## Positivas

- Achei ótimo o produto, estão de parabéns!
- Comprei por um valor barato. Maravilhoso.
- O custo era barato, mas era defeituoso. Se você tiver sorte, vale a pena.

## Negativas

- Lixo
- Simplesmente não vale a pena comprar
- Caro demais
- Compre se quiser ter dor de cabeça! Defeituoso!
- Produto muito ruim! Eu não compro mais nesta loja, a entrega atrasou e custou muito dinheiro.
