import pandas as pd

# Define the 50 Freud dream interpretations with structured formatting
freud_dreams = [
    ("Falling from a great height", "Dream: Falling from a great height\nInterpretation: To dream that you are falling signifies fear of losing control or failing in life."),
    ("Being naked in public", "Dream: Being naked in public\nInterpretation: To see that you are naked in public suggests feelings of vulnerability or exposure."),
    ("Flying", "Dream: Flying\nInterpretation: Dreaming of flying symbolizes a desire for freedom or escape from constraints."),
    ("Teeth falling out", "Dream: Teeth falling out\nInterpretation: If you dream about losing your teeth, it represents anxiety about appearance or communication."),
    ("Death", "Dream: Death\nInterpretation: To dream that you or someone else has died symbolizes change or transition rather than literal death."),
    ("Exams or tests", "Dream: Exams or tests\nInterpretation: Seeing yourself taking an exam in a dream reflects feelings of being judged or evaluated in waking life."),
    ("Being chased", "Dream: Being chased\nInterpretation: To dream that you are being chased indicates avoiding a problem or fear in waking life."),
    ("Drowning", "Dream: Drowning\nInterpretation: If you see yourself drowning, it may symbolize overwhelming emotions or a sense of helplessness."),
    ("Being late", "Dream: Being late\nInterpretation: Dreaming of being late suggests stress or fear of missing out on opportunities."),
    ("Being lost", "Dream: Being lost\nInterpretation: To see that you are lost in a dream represents confusion, uncertainty, or a lack of direction in life."),
    ("Driving an out-of-control vehicle", "Dream: Driving an out-of-control vehicle\nInterpretation: To dream that you are driving an uncontrollable vehicle signifies fears of losing control in real life."),
    ("Meeting a deceased loved one", "Dream: Meeting a deceased loved one\nInterpretation: Seeing a deceased loved one in a dream could signify unresolved emotions or a need for closure."),
    ("Pregnancy", "Dream: Pregnancy\nInterpretation: Dreaming of pregnancy symbolizes creativity, new beginnings, or personal growth."),
    ("Being unable to speak", "Dream: Being unable to speak\nInterpretation: If you dream about losing your voice, it may indicate feelings of repression or being unheard."),
    ("Fire", "Dream: Fire\nInterpretation: To dream that something is on fire represents passion, transformation, or destruction."),
    ("Water", "Dream: Water\nInterpretation: Seeing water in a dream often reflects emotions, subconscious thoughts, or cleansing."),
    ("Animals attacking", "Dream: Animals attacking\nInterpretation: To dream about animals attacking signifies inner fears, instincts, or unresolved conflicts."),
    ("Being trapped", "Dream: Being trapped\nInterpretation: Dreaming of being trapped could symbolize feeling stuck in a situation or relationship."),
    ("Finding hidden rooms", "Dream: Finding hidden rooms\nInterpretation: To see hidden rooms in a dream suggests self-discovery or unlocking new potential."),
    ("Losing something valuable", "Dream: Losing something valuable\nInterpretation: Dreaming of losing an object signifies anxiety about loss or personal insecurity."),
    ("Shadows or dark figures", "Dream: Shadows or dark figures\nInterpretation: To dream that you are surrounded by shadows symbolizes repressed emotions, fears, or the unknown."),
    ("Being in an old house", "Dream: Being in an old house\nInterpretation: If you dream of an old house, it represents revisiting past experiences or memories."),
    ("Climbing stairs", "Dream: Climbing stairs\nInterpretation: To dream that you are climbing stairs signifies progress, achievement, or spiritual growth."),
    ("Descending stairs", "Dream: Descending stairs\nInterpretation: Dreaming of going down stairs may represent setbacks, regression, or going deeper into the subconscious."),
    ("Mirrors", "Dream: Mirrors\nInterpretation: To see yourself in a mirror signifies self-reflection, identity, or personal perception."),
    ("Storms or tornadoes", "Dream: Storms or tornadoes\nInterpretation: If you dream about a storm, it reflects emotional turmoil, chaos, or sudden change."),
    ("Earthquakes", "Dream: Earthquakes\nInterpretation: Dreaming of an earthquake may symbolize instability, life changes, or insecurity."),
    ("Seeing someone crying", "Dream: Seeing someone crying\nInterpretation: To see someone crying in a dream indicates empathy, guilt, or emotional awareness."),
    ("Laughter", "Dream: Laughter\nInterpretation: Dreaming of laughter represents joy, release, or covering up deep emotions."),
    ("Being in a hospital", "Dream: Being in a hospital\nInterpretation: If you dream about being in a hospital, it signifies healing, self-care, or unresolved health concerns."),
    ("Food", "Dream: Food\nInterpretation: To dream about eating food represents nourishment, emotional satisfaction, or indulgence."),
    ("Wearing the wrong clothes", "Dream: Wearing the wrong clothes\nInterpretation: Seeing yourself in inappropriate clothing signifies insecurity, embarrassment, or feeling out of place."),
    ("Missing a flight or train", "Dream: Missing a flight or train\nInterpretation: Dreaming of missing a flight reflects fear of missed opportunities or time anxiety."),
    ("An intruder", "Dream: An intruder\nInterpretation: If you dream of an intruder in your home, it may symbolize external threats, fears, or self-doubt."),
    ("Childhood homes", "Dream: Childhood homes\nInterpretation: To dream of your childhood home represents nostalgia, unresolved childhood issues, or comfort."),
    ("Rain", "Dream: Rain\nInterpretation: Seeing rain in a dream signifies sadness, renewal, or emotional cleansing."),
    ("Bright light", "Dream: Bright light\nInterpretation: To dream of a bright light represents enlightenment, guidance, or hope."),
    ("Tunnels", "Dream: Tunnels\nInterpretation: Dreaming of a tunnel signifies transitions, birth, or searching for a way out."),
    ("Ghosts", "Dream: Ghosts\nInterpretation: Seeing ghosts in a dream may represent the past, unresolved issues, or hidden fears."),
    ("Doors", "Dream: Doors\nInterpretation: Dreaming of doors symbolizes opportunities, choices, or transitions in life."),
    ("Someone disappearing", "Dream: Someone disappearing\nInterpretation: To dream that someone disappears signifies loss, abandonment fears, or relationship concerns."),
]

# Convert to a DataFrame with two columns
df_freud_mixed = pd.DataFrame(freud_dreams, columns=["Dream", "Freudian Interpretation"])

# Save the dataset in training format with two columns
file_name = "freud_50_dreams.csv"
df_freud_mixed.to_csv(file_name, index=False)

print(f"✅ File '{file_name}' has been created successfully in your current directory.")
