# -*- coding: utf-8 -*-
import requests

#resp = requests.post("http://localhost:5000/predict", json={"raw_text":"how do you stop war?"})

# resp_prod = requests.post("http://213.159.215.173:5000/get_summary", json={"raw_text":"A significant number of executives from 151 financial institutions in 33 countries say that within the next two years they expect to become mass adopters of AI and expect AI to become an essential business driver across the financial industry."})
resp_prod = requests.post("http://35.202.164.44:5000/get_summary", json={"raw_text":"A significant number of executives from 151 financial institutions in 33 countries say that within the next two years they expect to become mass adopters of AI and expect AI to become an essential business driver across the financial industry."})

#print(resp.json())
#print(str(resp))

print('prod:', resp_prod.json())
print('prod:', str(resp_prod))