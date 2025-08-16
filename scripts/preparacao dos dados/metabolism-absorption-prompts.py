import xml.etree.ElementTree as ET
import json
import random

output_path = "/kaggle/working/metabolism_absorption.json"
metabolism_absorption_path = "/kaggle/input/metabolism-absorption-xml/metabolism_absorption.xml"
negative_input_path = "/kaggle/input/negative-examples/negative_examples.txt"

tree = ET.parse(metabolism_absorption_path)
root = tree.getroot()

print(f"Total de drogas: {len(root.findall('drug'))}")

entrys = []
with open(output_path, 'w', encoding='utf-8') as f:
    for drug in root.findall('drug'):
        name = drug.find('name').text
        
        if drug.find('metabolism') != None:
            metabolism = drug.find('metabolism').text
            metabolism = metabolism.replace("\n", "")
            metabolism = " ".join(metabolism.split())
            questions_meta = [
                f"What is the metabolism of {name} like?",
                f"How is {name} metabolized?",
                f"What is the metabolism process of {name}?",
                f"Can you explain how {name} is metabolized in the body?",
                f"In what way is {name} metabolized?",
                f"Describe the metabolic pathway of {name}.",
                f"What happens during the metabolism of {name}?",
                f"How does the body process {name} metabolically?",
                f"Tell me about {name}'s metabolism mechanism."
            ]
            for i in range(4):    
                user_meta = random.choice(questions_meta)
                questions_meta.remove(user_meta)
                entry_meta = {
                    "instruction": f"{user_meta}",
                    "input": f"{name}",
                    "output": f"{metabolism}"
                }
                entrys.append(entry_meta)

        
        if drug.find('absorption') != None:
            absorption = drug.find('absorption').text
            absorption = absorption.replace("\n", "")
            absorption = " ".join(absorption.split())
            questions_abs = [
                f"What is the absorption of {name} like?",
                f"How is {name} absorbed?",
                f"What is the absorption process of {name}?",
                f"Can you explain how {name} is absorbed in the body?",
                f"In what way is {name} absorbed?",
                f"Describe the absorption characteristics of {name}.",
                f"What happens during the absorption of {name}?",
                f"How does the body absorb {name}?",
                f"Tell me about {name}'s absorption mechanism."
            ]
            for i in range(4):
                user_abs = random.choice(questions_abs)
                questions_abs.remove(user_abs)
                entry_abs = {
                    "instruction": f"{user_abs}",
                    "input": f"{name}",
                    "output": f"{absorption}"
                }
                entrys.append(entry_abs)

    with open(negative_input_path, 'r', encoding='utf-8') as negative_ex:
        for line in negative_ex:
            clean_line = line.strip()
            if clean_line:  
                entry_negative = {
                    "instruction": line, 
                    "input": "", 
                    "output": "I can't answer that."
                }
                entrys.append(entry_negative)
    
    f.write(json.dumps(entrys, ensure_ascii=False))