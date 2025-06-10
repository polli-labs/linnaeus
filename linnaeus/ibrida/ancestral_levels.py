# linnaeus/ibrida/ancestral_levels.py
### COPIED FROM ibrida.utils.schemas.ancestral_levels


class AncestralLevels:
    def __init__(self):
        self.name_to_level = {
            "subspecies": 5,
            "species": 10,
            "complex": 11,
            "subsection": 12,
            "section": 13,
            "subgenus": 15,
            "genus": 20,
            "subtribe": 24,
            "tribe": 25,
            "supertribe": 26,
            "subfamily": 27,
            "family": 30,
            "epifamily": 32,
            "superfamily": 33,
            "zoosubsection": 33.5,
            "zoosection": 34,
            "parvorder": 34.5,
            "infraorder": 35,
            "suborder": 37,
            "order": 40,
            "superorder": 43,
            "subterclass": 44,
            "infraclass": 45,
            "subclass": 47,
            "class": 50,
            "superclass": 53,
            "subphylum": 57,
            "phylum": 60,
            "subkingdom": 67,
            "kingdom": 70,
            "stateofmatter": 100,
        }
        self.level_to_name = {v: k for k, v in self.name_to_level.items()}

    def format_level(self, level):
        if isinstance(level, str):
            level = level.replace("L", "").replace("l", "").replace("_", ".")
            if level.isdigit() or "." in level:
                level = float(level) if "." in level else int(level)
            else:
                return self.name_to_level.get(level, None)
        return level

    def get_level_name(self, level):
        level = self.format_level(level)
        return self.level_to_name.get(level, None)

    def get_level_str(self, level):
        level = self.format_level(level)
        return str(level).replace(".", "_")

    def get_level_Lstr(self, level, lowercase=False):
        level_str = self.get_level_str(level)
        return f"l{level_str}" if lowercase else f"L{level_str}"

    def as_names(self, skip_half_levels=False, primary_only=False, levels=None):
        levels = self.as_numbers(skip_half_levels, primary_only, levels)
        return [self.level_to_name[level] for level in levels]

    def as_numbers(self, skip_half_levels=False, primary_only=False, levels=None):
        if levels is None:
            levels = sorted(self.name_to_level.values())
        if skip_half_levels:
            levels = [level for level in levels if level == int(level)]
        if primary_only:
            levels = [level for level in levels if level % 10 == 0]
        return levels

    def as_strings(self, skip_half_levels=False, primary_only=False, levels=None):
        levels = self.as_numbers(skip_half_levels, primary_only, levels)
        return [self.get_level_str(level) for level in levels]

    def as_Lstrings(
        self, skip_half_levels=False, primary_only=False, lowercase=False, levels=None
    ):
        levels = self.as_numbers(skip_half_levels, primary_only, levels)
        return [self.get_level_Lstr(level, lowercase) for level in levels]

    def get_taxon_id_colname(self, level, schema):
        if level is None:
            return self.get_all_taxon_id_colnames(
                schema, skip_half_levels=False, primary_only=False
            )
        level_str = self.get_level_str(level)
        if schema == "taxaDB_sql":
            return f"L{level_str}_taxon_id"
        elif schema == "taxaDB_redis":
            return f"L{level_str}"  # Redis key for the level
        elif schema == "metadata_expanded":
            return f"L{level_str}_taxonID"
        else:
            raise ValueError(f"Unknown schema: {schema}")

    def get_name_colname(self, level, schema):
        if level is None:
            return self.get_all_name_colnames(
                schema, skip_half_levels=False, primary_only=False
            )
        level_str = self.get_level_str(level)
        if schema == "taxaDB_sql":
            return f"L{level_str}_name"
        elif schema == "taxaDB_redis":
            return f"L{level_str}"  # Redis key for the level
        elif schema == "metadata_expanded":
            return f"L{level_str}_name"
        else:
            raise ValueError(f"Unknown schema: {schema}")

    def get_all_taxon_id_colnames(
        self, schema, skip_half_levels=False, primary_only=False, levels=None
    ):
        levels = self.as_numbers(skip_half_levels, primary_only, levels)
        colnames = []
        for level in levels:
            level_str = self.get_level_str(level)
            if schema == "taxaDB_sql":
                colnames.append(f"L{level_str}_taxon_id")
            elif schema == "taxaDB_redis":
                colnames.append(f"L{level_str}")  # Redis key for the level
            elif schema == "metadata_expanded":
                colnames.append(f"L{level_str}_taxonID")
            else:
                raise ValueError(f"Unknown schema: {schema}")
        return colnames

    def get_all_name_colnames(
        self, schema, skip_half_levels=False, primary_only=False, levels=None
    ):
        levels = self.as_numbers(skip_half_levels, primary_only, levels)
        colnames = []
        for level in levels:
            level_str = self.get_level_str(level)
            if schema == "taxaDB_sql":
                colnames.append(f"L{level_str}_name")
            elif schema == "taxaDB_redis":
                colnames.append(f"L{level_str}")  # Redis key for the level
            elif schema == "metadata_expanded":
                colnames.append(f"L{level_str}_name")
            else:
                raise ValueError(f"Unknown schema: {schema}")
        return colnames
