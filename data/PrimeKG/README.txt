Please download the raw PrimeKG data from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM

PrimeKG

Here, we present the Precision Medicine Knowledge Graph (PrimeKG). This resource provides a holistic view of diseases. We have integrated 20 high-quality datasets, biorepositories and ontologies to curate this knowledge graph. PrimeKG systematically captures information about 17,080 diseases with 4,050,249 relationships representing various major biological scales, including diseases, drugs, genes, proteins, exposures, phenotypes, drug side effects, molecular functions, cellular components, biological processes, anatomical regions, and pathways. Disease nodes in our multi-relational knowledge graph are densely connected to every other node type. PrimeKG's rich graph structure is supplemented with textual descriptions of clinical guidelines for drug and disease nodes to enable multi-modal disease exploration. For further details, please read our publication cited below. To get started with using PrimeKG, please explore the tutorial shared on our GitHub repository: https://github.com/mims-harvard/PrimeKG

Github repository link: https://github.com/mims-harvard/PrimeKG

Description of PrimeKG 

-----------------------------------------------------------------------------------------------
Filename				Description
-----------------------------------------------------------------------------------------------
nodes.csv				Contains node level information
					Primary key: `node_index`
-----------------------------------------------------------------------------------------------
edges.csv				Contains undirected relationships between nodes 
					Primary key: (`x_index`, `y_index`)
-----------------------------------------------------------------------------------------------
kg.csv					This is the Precision Medicine knowledge graph  
					Primary key: (`x_index`, `y_index`)
-----------------------------------------------------------------------------------------------
disease_features.csv			Contains textual descriptions of diseases 
					Primary key: `node_index`
-----------------------------------------------------------------------------------------------
drug_features.csv			Contains textual descriptions of diseases 
					Primary key: `node_index`
-----------------------------------------------------------------------------------------------
kg_raw.csv 				Intermediate PrimeKG made by joining nodes and edges
-----------------------------------------------------------------------------------------------
kg_giant.csv				Intermediate PrimeKG made by taking LCC of kg_raw.csv -----------------------------------------------------------------------------------------------
kg_grouped.csv				Intermediate PrimeKG made by grouping diseases  -----------------------------------------------------------------------------------------------
kg_grouped_diseases.csv			List of all diseases and their assigned group name  
-----------------------------------------------------------------------------------------------
kg_grouped_diseases_bert_map.csv	Manual grouping created for diseases using BERT model
-----------------------------------------------------------------------------------------------


