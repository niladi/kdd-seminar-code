# %%

from models.clit_mock import Graph, IntersectionNode, MajorityVoting, UnionNode
from config import Config
from data.dataset import ClitResultDataset
import torch
from transformers import LongformerModel, LongformerTokenizer
from tqdm.auto import tqdm
import itertools

from util import flat_map

# %%

# Initialize the model and tokenizer
model_name = "allenai/longformer-base-4096"
model = LongformerModel.from_pretrained(model_name)
tokenizer = LongformerTokenizer.from_pretrained(model_name)

# Tokenize input text
text = "ATHLETICS - BRUSSELS GRAND PRIX RESULTS .  BRUSSELS 1996-08-23  Leading results in the Brussels Grand Prix athletics meeting on Friday :  Women 's discus  1. Ilke Wyludda ( Germany ) 66.60 metres  2. Ellina Zvereva ( Belarus ) 65.66  3. Franka Dietzsch ( Germany ) 61.74  4. Natalya Sadova ( Russia ) 61.64  5. Mette Bergmann ( Norway ) 61.44  6. Nicoleta Grasu ( Romania ) 61.36  7. Olga Chernyavskaya ( Russia ) 60.46  8. Irina Yatchenko ( Belarus ) 58.92  Women 's 100 metres hurdles  1. Ludmila Engquist ( Sweden ) 12.60 seconds  2. Michelle Freeman ( Jamaica ) 12.77  3. Aliuska Lopez ( Cuba ) 12.85  4. Dionne Rose ( Jamaica ) 12.88  5. Brigita Bukovec ( Slovakia ) 12.95  6. Yulia Graudin ( Russia ) 12.96  7. Julie Baumann ( Switzerland ) 13.36  8. Patricia Girard-Leno ( France ) 13.36  9. Dawn Bowles ( U.S. ) 13.53  Men 's 110 metres hurdles  1. Allen Johnson ( U.S. ) 12.92 seconds  2. Colin Jackson ( Britain ) 13.24  3. Emilio Valle ( Cuba ) 13.33  4. Sven Pieters ( Belgium ) 13.37  5. Steve Brown ( U.S. ) 13.38  6. Frank Asselman ( Belgium ) 13.64  7. Hubert Grossard ( Belgium ) 13.65  8. Jonathan N'Senga ( Belgium ) 13.66  9. Johan Lisabeth ( Belgium ) 13.75  Women 's 5,000 metres  1. Roberta Brunet ( Italy ) 14 minutes 48.96 seconds  2. Fernanda Ribeiro ( Portugal ) 14:49.81  3. Sally Barsosio ( Kenya ) 14:58.29  4. Paula Radcliffe ( Britain ) 14:59.70  5. Julia Vaquero ( Spain ) 15:04.94  6. Catherine McKiernan ( Ireland ) 15:07.57  7. Annette Peters ( U.S. ) 15:07.85  8. Pauline Konga ( Kenya ) 15:11.40  Men 's 100 metres  1. Dennis Mitchell ( U.S. ) 10.03 seconds  2. Donovan Bailey ( Canada ) 10.09  3. Carl Lewis ( U.S. ) 10.10  4. Ato Boldon ( Trinidad ) 10.12  5. Linford Christie ( Britain ) 10.14  6. Davidson Ezinwa ( Nigeria ) 10.15  7. Jon Drummond ( U.S. ) 10.16  8. Bruny Surin ( Canada ) 10.30  Men 's 400 metres hurdles  1. Derrick Adkins ( U.S. ) 47.93 seconds  2. Samuel Matete ( Zambia ) 47.99  3. Rohan Robinson ( Australia ) 48.86  4. Torrance Zellner ( U.S. ) 49.06  5. Jean-Paul Bruwier ( Belgium ) 49.24  6. Dusan Kovacs ( Hungary ) 49.31  7. Calvin Davis ( U.S. ) 49.49  8. Laurent Ottoz ( Italy ) 49.61  9. Marc Dollendorf ( Belgium ) 50.36  Women 's 100 metres  1. Gail Devers ( U.S. ) 10.84 seconds  2. Gwen Torrence ( U.S. ) 11.00  3. Merlene Ottey ( Jamaica ) 11.04  4. Mary Onyali ( Nigeria ) 11.09  5. Chryste Gaines ( U.S. ) 11.18  6. Zhanna Pintusevich ( Ukraine ) 11.27  7. Irina Privalova ( Russia ) 11.28  8. Natalia Voronova ( Russia ) 11.28  9. Juliet Cuthbert ( Jamaica ) 11.31  Women 's 1,500 metres  1. Regina Jacobs ( U.S. ) 4 minutes 01.77 seconds  2. Patricia Djate ( France ) 4:02.26  3. Carla Sacramento ( Portugal ) 4:02.67  4. Yekaterina Podkopayeva ( Russia ) 4:04.78  5. Margret Crowley ( Australia ) 4:05.00  6. Leah Pells ( Canada ) 4:05.64  7. Sarah Thorsett ( U.S. ) 4:06.80  8. Sinead Delahunty ( Ireland ) 4:07.27  3,000 metres steeplechase  1. Joseph Keter ( Kenya ) 8 minutes 10.02 seconds  2. Patrick Sang ( Kenya ) 8:12.04  3. Moses Kiptanui ( Kenya ) 8:12.65  4. Gideon Chirchir ( Kenya ) 8:15.69  5. Richard Kosgei ( Kenya ) 8:16.80  6. Larbi El Khattabi ( Morocco ) 8:17.29  7. Eliud Barngetuny ( Kenya ) 8:17.66  8. Bernard Barmasai ( Kenya ) 8:17.94  Men 's 400 metres  1. Michael Johnson ( U.S. ) 44.29 seconds  2. Derek Mills ( U.S. ) 44.78  3. Anthuan Maybank ( U.S. ) 44.92  4. Davis Kamoga ( Uganda ) 44.96  5. Jamie Baulch ( Britain ) 45.08  6. Sunday Bada ( Nigeria ) 45.21  7. Samson Kitur ( Kenya ) 45.34  8. Mark Richardson ( Britain ) 45.67  9. Jason Rouser ( U.S. ) 46.11  Men 's 200 metres  1. Frankie Fredericks ( Namibia ) 19.92 seconds  2. Ato Boldon ( Trinidad ) 19.99  3. Jeff Williams ( U.S. ) 20.21  4. Jon Drummond ( U.S. ) 20.42  5. Patrick Stevens ( Belgium ) 20.42  6. Michael Marsh ( U.S. ) 20.43  7. Ivan Garcia ( Cuba ) 20.45  8. Eric Wymeersch ( Belgium ) 20.84  9. Lamont Smith ( U.S. ) 21.08  Women 's 1,000 metres  1. Svetlana Masterkova ( Russia ) 2 minutes 28.98 seconds  ( world record )  2. Maria Mutola ( Mozambique ) 2:29.66  3. Malgorzata Rydz ( Poland ) 2:39.00  4. Anja Smolders ( Belgium ) 2:43.06  5. Veerle De Jaeghere ( Belgium ) 2:43.18  6. Eleonora Berlanda ( Italy ) 2:43.44  7. Anneke Matthijs ( Belgium ) 2:43.82  8. Jacqueline Martin ( Spain ) 2:44.22  Women 's 200 metres  1. Mary Onyali ( Nigeria ) 22.42 seconds  2. Inger Miller ( U.S. ) 22.66  3. Irina Privalova ( Russia ) 22.68  4. Natalia Voronova ( Russia ) 22.73  5. Marina Trandenkova ( Russia ) 22.84  6. Chandra Sturrup ( Bahamas ) 22.85  7. Zundra Feagin ( U.S. ) 23.18  8. Galina Malchugina ( Russia ) 23.25  Women 's 400 metres  1. Cathy Freeman ( Australia ) 49.48 seconds  2. Marie-Jose Perec ( France ) 49.72  3. Falilat Ogunkoya ( Nigeria ) 49.97  4. Pauline Davis ( Bahamas ) 50.14  5. Fatima Yussuf ( Nigeria ) 50.14  6. Maicel Malone ( U.S. ) 50.51  7. Hana Benesova ( Czech Republic ) 51.71  8. Ann Mercken ( Belgium ) 53.55  Men 's 3,000 metres  1. Daniel Komen ( Kenya ) 7 minutes 25.87 seconds  2. Khalid Boulami ( Morocco ) 7:31.65  3. Bob Kennedy ( U.S. ) 7:31.69  4. El Hassane Lahssini ( Morocco ) 7:32.44  5. Thomas Nyariki ( Kenya ) 7:35.56  6. Noureddine Morceli ( Algeria ) 7:36.81  7. Fita Bayesa ( Ethiopia ) 7:38.09  8. Martin Keino ( Kenya ) 7:38.88  Men 's discus  1. Lars Riedel ( Germany ) 66.74 metres  2. Anthony Washington ( U.S. ) 66.72  3. Vladimir Dubrovshchik ( Belarus ) 64.02  4. Virgilius Alekna ( Lithuania ) 63.62  5. Juergen Schult ( Germany ) 63.48  6. Vassiliy Kaptyukh ( Belarus ) 61.80  7. Vaclavas Kidikas ( Lithuania ) 60.92  8. Michael Mollenbeck ( Germany ) 59.24  Men 's triple jump  1. Jonathan Edwards ( Britain ) 17.50 metres  2. Yoelvis Quesada ( Cuba ) 17.29  3. Brian Wellman ( Bermuda ) 17.05  4. Kenny Harrison ( U.S. ) 16.97  5. Gennadi Markov ( Russia ) 16.66  6. Francis Agyepong ( Britain ) 16.63  7. Rogel Nachum ( Israel ) 16.36  8. Sigurd Njerve ( Norway ) 16.35  Men 's 1,500 metres  1. Hicham El Guerrouj ( Morocco ) three minutes 29.05 seconds  2. Isaac Viciosa ( Spain ) 3:33.00  3. William Tanui ( Kenya ) 3:33.36  4. Elijah Maru ( Kenya ) 3:33.64  5. Marcus O'Sullivan ( Ireland ) 3:33.77  6. John Mayock ( Britain ) 3:33.94  7. Laban Rotich ( Kenya ) 3:34.12  8. Christophe Impens ( Belgium ) 3:34.13  Women 's high jump  1. Stefka Kostadinova ( Bulgaria ) 2.03 metres  2. Inga Babakova ( Ukraine ) 2.03  3. Alina Astafei ( Germany ) 1.97  4. Tatyana Motkova ( Russia ) 1.94  5. Svetlana Zalevskaya ( Kazakhstan ) 1.91  6. Yelena Gulyayeva ( Russia ) 1.88  7. Hanna Haugland ( Norway ) 1.88  8 equal .  Olga Boshova ( Moldova ) 1.85  8 equal .  Nele Zilinskiene ( Lithuania ) 1.85  Men 's 10,000 metres  1. Salah Hissou ( Morocco ) 26 minutes 38.08 seconds ( world  record )  2. Paul Tergat ( Kenya ) 26:54.41  3. Paul Koech ( Kenya ) 26:56.78  4. William Kiptum ( Kenya ) 27:18.84  5. Aloys Nizigama ( Burundi ) 27:25.13  6. Mathias Ntawulikura ( Rwanda ) 27:25.48  7. Abel Anton ( Spain ) 28:18.44  8. Kamiel Maase ( Netherlands ) 28.29.42  9. Worku Bekila ( Ethiopia ) 28.42.23  10. Robert Stefko ( Slovakia ) 28:42.26"
inputs = tokenizer(
    text, return_tensors="pt", max_length=4096, truncation=True, padding="max_length"
)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state


# %%

print(embeddings.shape)
# %%
# Generate a two-dimensional list
two_dim_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(two_dim_list[1])


# %%


test: int


def foo(test: int) -> int:
    return test


foo("baum")
# %%


# %%
