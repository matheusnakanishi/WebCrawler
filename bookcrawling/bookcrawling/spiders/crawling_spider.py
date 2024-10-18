from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
from scrapy.spiders import CrawlSpider, Rule
from sklearn.preprocessing import MinMaxScaler
from scrapy.linkextractors import LinkExtractor
import matplotlib.pyplot as plt

class BookRecommenderSpider(CrawlSpider):
    name = "myCrawler"
    allowed_domains = ["toscrape.com"]
    start_urls = ["http://books.toscrape.com/"]

    rules = (
        Rule(LinkExtractor(allow="catalogue/category"), follow=True),
        Rule(LinkExtractor(allow="catalogue", deny="category"), callback="parse_item"),
    )

    def __init__(self, *args, **kwargs):
        super(BookRecommenderSpider, self).__init__(*args, **kwargs)
        
        self.l_data = [
            {'price': 19.63, 'category': 'mystery'},
            {'price': 17.28, 'category': 'historical fiction'},
            {'price': 15.08, 'category': 'classics'},
            {'price': 15.97, 'category': 'romance'},
            {'price': 10.60, 'category': 'fiction'}
        ]

        self.dl_data = [
            {'price': 23.63, 'category': 'mystery'},
            {'price': 54.28, 'category': 'historical fiction'},
            {'price': 49.08, 'category': 'classics'},
            {'price': 35.97, 'category': 'romance'},
            {'price': 26.60, 'category': 'fiction'}
        ]
        
        # Combinar dados de l_data e dl_data para treino
        self.categories = {cat: i for i, cat in enumerate(set(b['category'] for b in (self.l_data + self.dl_data)))}
        self.model = self.train_model()

    def train_model(self):
        # Preparar os dados de treino
        X_liked = np.array([[book['price'], self.categories[book['category']]] for book in self.l_data])
        X_disliked = np.array([[book['price'], self.categories[book['category']]] for book in self.dl_data])

        # Normalizar os preços
        scaler = MinMaxScaler()
        X_liked[:, 0] = scaler.fit_transform(X_liked[:, [0]]).flatten()
        X_disliked[:, 0] = scaler.transform(X_disliked[:, [0]]).flatten()

        # Dados combinados
        X = np.vstack([X_liked, X_disliked])
        y = np.hstack([np.ones(len(X_liked)), np.zeros(len(X_disliked))])  # 1 para livros que o usuário gosta, 0 para livros que não gosta

        # Criar e treinar a árvore de decisão
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        self.scaler = scaler

        #self.plot_decision_tree(clf, X)

        return clf
    
    def plot_decision_tree(self, clf, X):
        # Desnormalizar os preços para exibição
        X_denormalized = X.copy()
        X_denormalized[:, 0] = self.scaler.inverse_transform(X[:, [0]]).flatten()

        # Plotar a árvore de decisão com os valores desnormalizados
        plt.figure(figsize=(12, 8))
        plot_tree(
            clf, 
            feature_names=["price", "category"], 
            class_names=["Dislike", "Like"], 
            filled=True, 
            rounded=True,
            impurity=False,
            proportion=False
        )
        plt.show()


    def predict_similarity(self, price, category):
        # Verificar a similaridade com os livros fornecidos
        category_value = self.categories.get(category.lower(), -1)
        if category_value == -1:
            return 0  # Se a categoria não existir, considere como irrelevante
        
        # Normalizar o preço de entrada
        price_normalized = self.scaler.transform([[price]])[0][0]

        features = np.array([[price_normalized, category_value]])
        return self.model.predict(features)[0]

    def parse_item(self, response):
        book_category = response.css(".breadcrumb li:nth-child(3) a::text").get().lower()
        price = response.css(".price_color::text").get()
        price_value = float(price.replace('£', '').strip())
        in_stock = "In stock" in response.css(".availability::text")[1].get().strip()

        # Verificar se o livro é semelhante com base no modelo treinado
        is_similar = self.predict_similarity(price_value, book_category)

        if is_similar == 1 and in_stock:
            yield {
                "title": response.css(".product_main h1::text").get(),
                "price": price_value,
                "link": response.url,
                "category": book_category,
                "availability": "In stock"
            }
