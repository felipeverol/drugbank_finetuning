import xml.etree.ElementTree as ET

# Função para processar cada <drug> e adicionar ao novo XML
def process_drug(drug, novo_root, namespaces):
    """
    Processa um elemento <drug> e adiciona as informações ao novo XML.
    """

    drug_name = drug.find('ns:name', namespaces).text
    drug_metabolism = drug.find('ns:metabolism', namespaces).text
    drug_absorption = drug.find('ns:absorption', namespaces).text

    if drug_metabolism or drug_absorption:
        novo_drug = ET.SubElement(novo_root, "drug", attrib=drug.attrib)
        ET.SubElement(novo_drug, "name").text = drug_name
        if drug_metabolism:
            ET.SubElement(novo_drug, "metabolism").text = drug_metabolism
        if drug_absorption:
            ET.SubElement(novo_drug, "absorption").text = drug_absorption
    

# Função principal para processar o arquivo XML
def process_xml(file_path, output_xml):
    """
    Processa um arquivo XML e salva os dados em um novo XML.
    """
    # Carregar o arquivo XML inteiro na memória
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Definir namespaces
    namespaces = {'ns': 'http://www.drugbank.ca'}
    
    # Criar a raiz do novo XML
    novo_root = ET.Element("drugs")
    
    # Processar cada <drug> dentro de <drugbank>
    for drug in root.findall('ns:drug', namespaces):
        process_drug(drug, novo_root, namespaces)
    
    # Salvar o novo XML em um arquivo
    novo_tree = ET.ElementTree(novo_root)
    novo_tree.write(output_xml, encoding='utf-8', xml_declaration=True)

# Caminho do arquivo XML de entrada
file_path = '/kaggle/input/fulldatabase/fulldatabase.xml'  # Substitua pelo caminho do seu arquivo

# Caminho do arquivo XML de saída
output_xml = '/kaggle/working/metabolism_absorption.xml'  # Nome do arquivo XML de saída

# Processar o arquivo XML e salvar os dados no novo XML
process_xml(file_path, output_xml)

print(f"Dados salvos em {output_xml}")