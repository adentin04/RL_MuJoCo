MuJoCo : Multi-Joint dynamics with Contact
Ce logiciel est pour tout ce qui concerne la recher sur ce qui demand la rapidité et précision de structures articulé
L'utilisateur defini les modeles dans le MJCF qui est "scene description language"
Les attributs du MJCF ont des valeurs par default
	ex: pour les objets le type par default est la sphere
Mujoco is used to train AI agents to perform complex, contact-rich behaviours


## Functions
XML = """ .... """

empty model = ``` <mujoco> <mujoco/> ``` 
<worldbudy> = tutti gli elementi fisici vivono qui dentro 
- top level body 
- global origin in cartesian coordinates
<geom name="red_box" type= "box" size=".2.2.2" rgba=1001 />

from_xml_string() = chiama il compilatore (modello) che crea anche una instance mjModel
mjModels= contiene la descrizione del modello e la sua grandezza fisica (sono valori fissi che non cambiano durante la simulazione) 

mjData= contiene lo stato e grandezze che non cambiano ogni passo.

data =mujoco.MjData(model) 
