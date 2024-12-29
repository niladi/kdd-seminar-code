import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Opel baut wieder ein kompaktes Cabrio. Rüsselsheim (dpa/tmn) - Kaum hat VW das neue Golf Cabrio enthüllt, zieht Opel jetzt nach: Offenbar als Reaktion auf die Premiere in Wolfsburg hat der Rüsselsheimer Hersteller den Bau eines neuen Open-Air-Modells bestätigt. Rüsselsheim (dpa/tmn) - Kaum hat VW das neue Golf Cabrio enthüllt, zieht Opel jetzt nach: Offenbar als Reaktion auf die Premiere in Wolfsburg hat der Rüsselsheimer Hersteller den Bau eines neuen Open-Air-Modells bestätigt. Das Cabrio basiert wieder auf dem Astra und soll im Jahr 2013 auf den Markt kommen, teilte das Unternehmen mit. Gebaut wird der Wagen mit weiteren Astra-Varianten in Gliwice in Polen. Das letzte Astra Cabrio war das Modell TwinTop mit einem versenkbaren Stahldach, das mit dem Generationswechsel in der Kompaktklasse vor rund einem Jahr eingestellt wurde. Angesichts der damaligen Unternehmenskrise hatten die Hessen zunächst offengelassen, ob sie diese Nische wieder besetzen würden. «Das neue Cabrio ist eine ideale Ergänzung unseres Produktportfolios», erklärte Opel-Vorstandschef Nick Reilly: «Mit seinem eleganten Design steht es in der Tradition unserer offenen und sportlichen Fahrzeuge und wird einen wichtigen positiven Effekt auf unsere Marke haben», so R"

# Process the text with the spacy pipeline
doc = nlp(text)

# Extract and print mentions
for ent in doc.ents:
    print(ent.text, ent.label_)
