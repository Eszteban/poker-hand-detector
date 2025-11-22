# Kártyalapok felismerése: Kódleírás és kielemzés

## Algoritmus leírása

- Élek keresése (Canny éldetektorral)
- Sarkok keresése, téglalap keresése
- Kivágás, perspektívikus torzítás (nem érzékeny az elhelyezkedésre)
- Bal felső sarok levágása
- Binarizálás
- Betű és szimbólum kinyerése
- Template matching
- Kinyert adatokkal ellenőrizni, van-e pókerkéz

## Eredmények kielemzése

Ezen projektnél cél volt, hogy ne használjunk neurális hálót, egyszerű eszközökkel próbáljuk meg megoldani. Ennek megfelelően csak OpenCV, Numpy és Matplotlib (megjelenítéshez) került felhasználásra.

Az elveknek viszont megvannak a korlátai és kompromisszumai: A kód csak akkor ismer fel 2 kártyát, ha a lapok nem fedik egymást. Ezt igyekeztünk feloldani, viszont nem jártunk sikerrel - erről írnék most.

Kettő, nagyjából hasonló ötlet merült fel. A kártyajátékoknál mikor tartjuk a lapokat, akkor a lapok egy sarkát tartjuk és mint egy legyező, látszanak a kártya átellenes sarkai. Amennyiben derékszögeket tudnánk találni, tudunk értékes információt kinyerni, a lapok sarkaival pedig az algoritmus többi része futtatható lenne.

Végül ezen kísérlet kudarcba zajlott: A kártyák egyszínűek és emiatt a kód nem talált derékszöget - csak a teljesen látszó lapot ismerte fel. Így ezen ág elvetésre került.

