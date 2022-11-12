def is_latin1(name):
    try:
        name.encode("latin-1")
        return True
    except:
        return False

date_of_beginning = {qid: int(year.strip()) for qid, year in (x.split("\t") for x in open("extras/neckar/entities.dates.tsv"))}

bad_orgs = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", 
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
bad_orgs = set(map(str.lower, bad_orgs))

builds = set()

from tqdm import tqdm
type_entities = type_entities = {"PER" : [], "ORG": [], "LOC" : [], "MISC": []}
for line in tqdm(open("extras/neckar/wikidata.entities")):
    t, qid, name, alias = line.strip('\n').split("\t")
    if qid in date_of_beginning and date_of_beginning[qid] >= 1999:
        continue
    aliases = alias.strip().split("||")
    if aliases[-1] == '':
        aliases = []
    aliases.append(name)
    if t == "PER":
        name_parts = name.strip().split(" ")
        if len(name_parts) > 1:
            try :
                build = name_parts[0][0] + '. ' + name_parts[-1]
            except:
                print(name_parts)
                break 
            aliases.append(build)
            builds.add(build)
    aliases = list(set(aliases))
    type_entities[t].extend(aliases)

set_gazette = {k: {x.lower() for x in set(v) if is_latin1(x) and x.lower() not in bad_orgs} for k, v in type_entities.items()}
print({k:len(v) for k, v in set_gazette.items()})