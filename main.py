
class AdvancedSchemaToGraphConverter:
    def __init__(self):
        self.graph = nx.Graph()
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Define patterns for common SQL operations
        self.matcher.add("AGGREGATION", [[{"LOWER": {"IN": ["average", "avg", "sum", "count", "max", "min"]}}]])
        self.matcher.add("GROUPING", [[{"LOWER": "group"}, {"LOWER": "by"}]])
        self.matcher.add("ORDERING", [[{"LOWER": {"IN": ["order", "sort"]}}, {"LOWER": "by"}]])
        self.matcher.add("LIMIT", [[{"LOWER": {"IN": ["top", "bottom"]}}, {"POS": "NUM"}]])
        self.matcher.add("TIME_RANGE", [[{"LOWER": {"IN": ["last", "past"]}}, {"POS": "NUM"}, {"LOWER": {"IN": ["day", "week", "month", "year"]}}]])

        # Add patterns for specific business concepts
        self.matcher.add("CUSTOMER_RETENTION", [[{"LOWER": "customer"}, {"LOWER": "retention"}]])
        self.matcher.add("POPULAR_PRODUCTS", [[{"LOWER": "popular"}, {"LOWER": "products"}]])
        self.matcher.add("BOUGHT_TOGETHER", [[{"LOWER": "bought"}, {"LOWER": "together"}]])

    def load_schema(self, schema_json: str):
        schema = json.loads(schema_json)
        self._process_schema(schema)
        self._update_phrase_matcher()

    def _process_schema(self, schema: Dict[str, Any]):
        for table_name, table_info in schema.items():
            self._add_table_node(table_name, table_info)
            self._process_columns(table_name, table_info['columns'])
            self._process_relationships(table_name, table_info.get('relationships', []))

    def _add_table_node(self, table_name: str, table_info: Dict[str, Any]):
        self.graph.add_node(table_name, type='table', description=table_info.get('description', ''))

    def _process_columns(self, table_name: str, columns: Dict[str, Any]):
        for column_name, column_info in columns.items():
            column_node = f"{table_name}.{column_name}"
            self.graph.add_node(column_node, type='column', 
                                data_type=column_info['data_type'],
                                constraints=column_info.get('constraints', []))
            self.graph.add_edge(table_name, column_node, type='has_column')

    def _process_relationships(self, table_name: str, relationships: list):
        for relationship in relationships:
            related_table = relationship['related_table']
            self.graph.add_edge(table_name, related_table, type='related_to',
                                relationship_type=relationship['type'])

    def _update_phrase_matcher(self):
        patterns = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] in ['table', 'column']:
                patterns.append(self.nlp(node.lower()))
        self.phrase_matcher.add("SCHEMA_ELEMENTS", patterns)

    def advanced_question_processing(self, question: str) -> Dict[str, Any]:
        doc = self.nlp(question.lower())
        
        # Extract entities and operations
        entities = [ent.text for ent in doc.ents]
        operations = []
        for match_id, start, end in self.matcher(doc):
            operations.append(doc[start:end].text)
        
        # Get relevant tables and columns
        matches = self.phrase_matcher(doc)
        relevant_elements = [doc[start:end].text for _, start, end in matches]
        
        tables, columns = self._classify_relevant_elements(relevant_elements)
        
        # Expand tables and columns based on the question context
        expanded_tables, expanded_columns = self._expand_relevant_elements(tables, columns, doc)
        
        # Identify potential joins and related tables
        potential_joins = self.find_potential_joins(expanded_tables)
        
        return {
            "entities": entities,
            "operations": operations,
            "tables": expanded_tables,
            "columns": expanded_columns,
            "potential_joins": potential_joins
        }

    def _classify_relevant_elements(self, elements: List[str]) -> Tuple[List[str], List[str]]:
        tables = []
        columns = []
        for element in elements:
            if '.' in element:
                columns.append(element)
            else:
                tables.append(element)
        return tables, columns

    def _expand_relevant_elements(self, tables: List[str], columns: List[str], doc) -> Tuple[List[str], List[str]]:
        expanded_tables = set(tables)
        expanded_columns = set(columns)
        
        # Expand based on relationships
        for table in tables:
            expanded_tables.update(self.get_related_tables(table))
        
        # Expand based on common query patterns
        if any(token.text in ['total', 'sum', 'amount'] for token in doc):
            expanded_columns.add('orders.total_amount')
            expanded_tables.add('orders')
        
        if any(token.text in ['customer', 'user'] for token in doc):
            expanded_tables.add('users')
        
        if any(token.text in ['product', 'item'] for token in doc):
            expanded_tables.add('products')
            expanded_tables.add('order_items')
        
        if any(token.text in ['category'] for token in doc):
            expanded_columns.add('products.category')
        
        if any(token.text in ['bought', 'purchased', 'together'] for token in doc):
            expanded_tables.update(['orders', 'order_items', 'products'])
        
        # Expand columns for all tables
        for table in expanded_tables:
            expanded_columns.update(self.get_table_columns(table))
        
        return list(expanded_tables), list(expanded_columns)

    def get_related_tables(self, table: str) -> List[str]:
        if table not in self.graph:
            return []
        return [node for node in nx.neighbors(self.graph, table) 
                if self.graph.nodes[node]['type'] == 'table']

    def get_table_columns(self, table: str) -> List[str]:
        if table not in self.graph:
            return []
        return [node for node in nx.neighbors(self.graph, table) 
                if self.graph.nodes[node]['type'] == 'column']

    def find_potential_joins(self, tables: List[str]) -> List[tuple]:
        joins = []
        for i in range(len(tables)):
            for j in range(i+1, len(tables)):
                if tables[i] in self.graph and tables[j] in self.graph:
                    try:
                        path = nx.shortest_path(self.graph, tables[i], tables[j])
                        joins.append((tables[i], tables[j], path))
                    except nx.NetworkXNoPath:
                        continue
        return joins

    def visualize(self):
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        plt.figure(figsize=(12, 8))
        
        color_map = {
            'table': '#FF9999',
            'column': '#66B2FF',
        }
        
        for node_type in color_map:
            node_list = [node for node, data in self.graph.nodes(data=True) if data['type'] == node_type]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=node_list, node_color=color_map[node_type], node_size=3000, alpha=0.8)

        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
        
        labels = {node: node for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type.capitalize(),
                          markerfacecolor=color, markersize=10)
                          for node_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        plt.title("Database Schema Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        
schema_json = '''
{
    "users": {
        "description": "Store user information",
        "columns": {
            "id": {"data_type": "integer", "constraints": ["primary_key"]},
            "username": {"data_type": "varchar", "constraints": ["unique"]},
            "email": {"data_type": "varchar", "constraints": ["unique"]},
            "age": {"data_type": "integer"},
            "registration_date": {"data_type": "date"}
        },
        "relationships": [
            {"related_table": "orders", "type": "has_many"}
        ]
    },
    "orders": {
        "description": "Store order information",
        "columns": {
            "id": {"data_type": "integer", "constraints": ["primary_key"]},
            "user_id": {"data_type": "integer", "constraints": ["foreign_key"]},
            "total_amount": {"data_type": "decimal"},
            "date": {"data_type": "date"}
        },
        "relationships": [
            {"related_table": "users", "type": "belongs_to"},
            {"related_table": "order_items", "type": "has_many"}
        ]
    },
    "products": {
        "description": "Store product information",
        "columns": {
            "id": {"data_type": "integer", "constraints": ["primary_key"]},
            "name": {"data_type": "varchar"},
            "price": {"data_type": "decimal"},
            "category": {"data_type": "varchar"}
        },
        "relationships": [
            {"related_table": "order_items", "type": "has_many"}
        ]
    },
    "order_items": {
        "description": "Store items within each order",
        "columns": {
            "id": {"data_type": "integer", "constraints": ["primary_key"]},
            "order_id": {"data_type": "integer", "constraints": ["foreign_key"]},
            "product_id": {"data_type": "integer", "constraints": ["foreign_key"]},
            "quantity": {"data_type": "integer"}
        },
        "relationships": [
            {"related_table": "orders", "type": "belongs_to"},
            {"related_table": "products", "type": "belongs_to"}
        ]
    }
}
'''