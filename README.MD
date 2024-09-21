#Advanced Schema-to-Graph Converter
The Advanced Schema-to-Graph Converter is a Python-based tool designed to convert relational database schemas into graph structures. This enriched graph structure facilitates complex query handling in text-to-SQL applications by adding context and intelligence to the schema representation. It also allows visualization of the schema relationships, making it easier to understand and query.

#Features
Schema-to-Graph Conversion: Automatically converts database schema (tables, columns, relationships) into a graph network.
Enriched Metadata: Captures additional context like column constraints, statistical metadata, and functional dependencies.
Advanced Question Processing: Utilizes NLP with pre-built patterns to understand complex queries and map them to relevant tables and columns.
Automatic Query Expansion: Expands the scope of queries by identifying related tables and columns based on natural language input.
Potential Join Detection: Finds and suggests joins between tables based on schema relationships.
Visualization: Graphically displays the schema using NetworkX and Matplotlib.