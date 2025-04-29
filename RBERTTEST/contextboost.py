import psycopg2

from RBERTTEST.fillBD import connection


class ContextBoost:
    def __init__(self, final_scores, doc_words):
        self.__final_scores = final_scores
        self.__doc_words = doc_words
        self.__cursor = psycopg2.connect(host="localhost", database="BBK_index", user="postgres", password="Dima2003",
                                         port=5432).cursor()

    def getKeySet(self, path):
        self.__cursor.execute(f"SELECT * FROM keywords_bbk WHERE path = '{path}'")
        keyset = set()
        for row in self.__cursor.fetchall():
            keyset.add(row[1])
        connection.commit()
        return keyset

    def processingTop0(self):
        # 26 Науки о Земле
        if "26 Науки о Земле" in self.__final_scores:
            earth_science_keywords = self.getKeySet(26.)

            self.__cursor.execute(rf"""SELECT path FROM index_bbk WHERE path::text ~ '^26\.\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 3""")
            for row in self.__cursor.fetchall():
                earth_science_keywords = earth_science_keywords.union(self.getKeySet(row[0]))

            matches = len(self.__doc_words & earth_science_keywords)
            if matches >= 5:
                self.__final_scores["26 Науки о Земле"] *= 1.8
            elif matches >= 3:
                self.__final_scores["26 Науки о Земле"] *= 1.4

        # 22 Физико-математические науки
        if "22 Физико-математические науки" in self.__final_scores:
            physics_math_keywords = self.getKeySet(22.)

            self.__cursor.execute(rf"""SELECT path FROM index_bbk WHERE path::text ~ '^22\.\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 3""")
            for row in self.__cursor.fetchall():
                physics_math_keywords = physics_math_keywords.union(self.getKeySet(row[0]))

            matches = len(self.__doc_words & physics_math_keywords)
            if matches >= 5:
                self.__final_scores["22 Физико-математические науки"] *= 1.8
            elif matches >= 3:
                self.__final_scores["22 Физико-математические науки"] *= 1.4

        # 24 Химические науки
        if "24 Химические науки" in self.__final_scores:
            chemistry_keywords = self.getKeySet(24.)

            self.__cursor.execute(rf"""SELECT path FROM index_bbk WHERE path::text ~ '^24\.\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 3""")
            for row in self.__cursor.fetchall():
                chemistry_keywords = chemistry_keywords.union(self.getKeySet(row[0]))

            matches = len(self.__doc_words & chemistry_keywords)
            if matches >= 5:
                self.__final_scores["24 Химические науки"] *= 1.8
            elif matches >= 3:
                self.__final_scores["24 Химические науки"] *= 1.4

        # 28 Биологические науки
        if "28 Биологические науки" in self.__final_scores:
            biology_keywords = self.getKeySet(28.)
            self.__cursor.execute(rf"""SELECT path FROM index_bbk WHERE path::text ~ '^28\.\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 3""")
            for row in self.__cursor.fetchall():
                biology_keywords = biology_keywords.union(self.getKeySet(row[0]))
            matches = len(self.__doc_words & biology_keywords)
            if matches >= 5:
                self.__final_scores["28 Биологические науки"] *= 1.8
            elif matches >= 3:
                self.__final_scores["28 Биологические науки"] *= 1.4

    def processingTop1(self):
        if "28.4 Микробиология" in self.__final_scores:
            microbio_keywords = self.getKeySet(28.4)
            matches = len(self.__doc_words & microbio_keywords)
            if matches >= 3:
                self.__final_scores["28.4 Микробиология"] *= 1.5
            elif matches >= 1:
                self.__final_scores["28.4 Микробиология"] *= 1.2

        if "28.5 Ботаника" in self.__final_scores:
            botany_keywords = self.getKeySet(28.5)
            matches = len(self.__doc_words & botany_keywords)
            if matches >= 4:
                self.__final_scores["28.5 Ботаника"] *= 1.6
            elif matches >= 2:
                self.__final_scores["28.5 Ботаника"] *= 1.3
            elif matches >= 1:
                self.__final_scores["28.5 Ботаника"] *= 1.1

        if "28.0 Общая биология" in self.__final_scores:
            obbio_keywords  = self.getKeySet(28.0)
            matches = len(self.__doc_words & obbio_keywords)
            if matches >= 5:
                self.__final_scores["28.0 Общая биология"] *= 1.8
            elif matches >= 3:
                self.__final_scores["28.0 Общая биология"] *= 1.5
            elif matches >= 1:
                self.__final_scores["28.0 Общая биология"] *= 1.2

        if "28.1 Палеонтология" in self.__final_scores:
            pal_keywords = self.getKeySet(28.1)
            matches = len(self.__doc_words & pal_keywords)
            if matches >= 3:
                self.__final_scores["28.1 Палеонтология"] *= 1.6
            elif matches >= 1:
                self.__final_scores["28.1 Палеонтология"] *= 1.3

        if "28.3 Вирусология" in self.__final_scores:
            virus_keywords = self.getKeySet(28.3)
            matches = len(self.__doc_words & virus_keywords)
            if matches >= 3:
                self.__final_scores["28.3 Вирусология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.3 Вирусология"] *= 1.3

        if "28.6 Зоология" in self.__final_scores:
            zoo_keywords = self.getKeySet(28.6)
            matches = len(self.__doc_words & zoo_keywords)
            if matches >= 4:
                self.__final_scores["28.6 Зоология"] *= 1.6
            elif matches >= 2:
                self.__final_scores["28.6 Зоология"] *= 1.3

        if "28.7 Биология человека. Антропология" in self.__final_scores:
            chel_keywords = self.getKeySet(28.7)
            matches = len(self.__doc_words & chel_keywords)
            if matches >= 3:
                self.__final_scores["28.7 Биология человека. Антропология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.7 Биология человека. Антропология"] *= 1.3

    def processingTop2(self):
        if "28.00 Теоретическая биология" in self.__final_scores:
            teorbio_keywords = self.getKeySet(28.00)
            matches = len(self.__doc_words & teorbio_keywords)
            if matches >= 4:
                self.__final_scores["28.00 Теоретическая биология"] *= 1.7
            elif matches >= 2:
                self.__final_scores["28.00 Теоретическая биология"] *= 1.4

        if "28.01 Жизнь. Живые системы" in self.__final_scores:
            life_keywords = self.getKeySet(28.01)
            matches = len(self.__doc_words & life_keywords)
            if matches >= 5:
                self.__final_scores["28.01 Жизнь. Живые системы"] *= 1.8
            elif matches >= 3:
                self.__final_scores["28.01 Жизнь. Живые системы"] *= 1.5
            elif matches >= 1:
                self.__final_scores["28.01 Жизнь. Живые системы"] *= 1.2

        if "28.02 Эволюционная биология" in self.__final_scores:
            evobio_keywords = self.getKeySet(28.02)
            matches = len(self.__doc_words & evobio_keywords)
            if matches >= 5:
                self.__final_scores["28.02 Эволюционная биология"] *= 1.9
            elif matches >= 3:
                self.__final_scores["28.02 Эволюционная биология"] *= 1.6
            elif matches >= 1:
                self.__final_scores["28.02 Эволюционная биология"] *= 1.3

        if "28.04 Общая генетика" in self.__final_scores:
            genet_keywords = self.getKeySet(28.04)
            matches = len(self.__doc_words & genet_keywords)
            if matches >= 5:
                self.__final_scores["28.04 Общая генетика"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.04 Общая генетика"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.04 Общая генетика"] *= 1.4

        if "28.05 Общая цитология" in self.__final_scores:
            cytol_keywords = self.getKeySet(28.05)
            matches = len(self.__doc_words & cytol_keywords)
            if matches >= 5:
                self.__final_scores["28.05 Общая цитология"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.05 Общая цитология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.05 Общая цитология"] *= 1.4

        if "28.09 Общая систематика организмов" in self.__final_scores:
            syst_keywords = self.getKeySet(28.09)
            matches = len(self.__doc_words & syst_keywords)
            if matches >= 4:
                self.__final_scores["28.09 Общая систематика организмов"] *= 1.8
            elif matches >= 2:
                self.__final_scores["28.09 Общая систематика организмов"] *= 1.5

        if "28.12 Биостратиграфия" in self.__final_scores:
            biostrat_keywords = self.getKeySet(28.12)
            matches = len(self.__doc_words & biostrat_keywords)
            if matches >= 3:
                self.__final_scores["28.12 Биостратиграфия"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.12 Биостратиграфия"] *= 1.4

        if "28.14 Микропалеонтология" in self.__final_scores:
            micropal_keywords = self.getKeySet(28.14)
            matches = len(self.__doc_words & micropal_keywords)
            if matches >= 3:
                self.__final_scores["28.14 Микропалеонтология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.14 Микропалеонтология"] *= 1.4

        if "28.15 Палеоботаника" in self.__final_scores:
            paleobot_keywords = self.getKeySet(28.15)
            matches = len(self.__doc_words & paleobot_keywords)
            if matches >= 3:
                self.__final_scores["28.15 Палеоботаника"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.15 Палеоботаника"] *= 1.4

        if "28.31 Происхождение вирусов" in self.__final_scores:
            virus_origin_keywords = self.getKeySet(28.31)
            matches = len(self.__doc_words & virus_origin_keywords)
            if matches >= 3:
                self.__final_scores["28.31 Происхождение вирусов"] *= 1.8
            elif matches >= 1:
                self.__final_scores["28.31 Происхождение вирусов"] *= 1.5

        if "28.33 Жизненные циклы вирусов" in self.__final_scores:
            virus_life_keywords = self.getKeySet(28.33)
            matches = len(self.__doc_words & virus_life_keywords)
            if matches >= 4:
                self.__final_scores["28.33 Жизненные циклы вирусов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.33 Жизненные циклы вирусов"] *= 1.6

        if "28.34 Генетика вирусов" in self.__final_scores:
            virus_gen_keywords = self.getKeySet(28.34)
            matches = len(self.__doc_words & virus_gen_keywords)
            if matches >= 4:
                self.__final_scores["28.34 Генетика вирусов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.34 Генетика вирусов"] *= 1.6

        if "28.36 Морфология вирусов" in self.__final_scores:
            virus_morph_keywords = self.getKeySet(28.36)
            matches = len(self.__doc_words & virus_morph_keywords)
            if matches >= 3:
                self.__final_scores["28.36 Морфология вирусов"] *= 1.8
            elif matches >= 1:
                self.__final_scores["28.36 Морфология вирусов"] *= 1.5

        if "28.43 Биология развития микроорганизмов" in self.__final_scores:
            microb_dev_keywords = self.getKeySet(28.43)
            matches = len(self.__doc_words & microb_dev_keywords)
            if matches >= 4:
                self.__final_scores["28.43 Биология развития микроорганизмов"] *= 1.8
            elif matches >= 2:
                self.__final_scores["28.43 Биология развития микроорганизмов"] *= 1.5

        if "28.46 Морфология микроорганизмов" in self.__final_scores:
            microb_morph_keywords = self.getKeySet(28.46)
            matches = len(self.__doc_words & microb_morph_keywords)
            if matches >= 5:
                self.__final_scores["28.46 Морфология микроорганизмов"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.46 Морфология микроорганизмов"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.46 Морфология микроорганизмов"] *= 1.4

        if "28.39 Систематика вирусов" in self.__final_scores:
            virus_syst_keywords = self.getKeySet(28.39)
            matches = len(self.__doc_words & virus_syst_keywords)
            if matches >= 4:
                self.__final_scores["28.39 Систематика вирусов"] *= 1.8
            elif matches >= 2:
                self.__final_scores["28.39 Систематика вирусов"] *= 1.5

        if "28.41 Происхождение микроорганизмов" in self.__final_scores:
            microb_origin_keywords = self.getKeySet(28.41)
            matches = len(self.__doc_words & microb_origin_keywords)
            if matches >= 3:
                self.__final_scores["28.41 Происхождение микроорганизмов"] *= 1.8
            elif matches >= 1:
                self.__final_scores["28.41 Происхождение микроорганизмов"] *= 1.5

        if "28.44 Генетика микроорганизмов" in self.__final_scores:
            microb_gen_keywords = self.getKeySet(28.44)
            matches = len(self.__doc_words & microb_gen_keywords)
            if matches >= 5:
                self.__final_scores["28.44 Генетика микроорганизмов"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.44 Генетика микроорганизмов"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.44 Генетика микроорганизмов"] *= 1.4

        if "28.47 Физико-химическая биология микроорганизмов" in self.__final_scores:
            microb_phys_keywords = self.getKeySet(28.47)
            matches = len(self.__doc_words & microb_phys_keywords)
            if matches >= 5:
                self.__final_scores["28.47 Физико-химическая биология микроорганизмов"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.47 Физико-химическая биология микроорганизмов"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.47 Физико-химическая биология микроорганизмов"] *= 1.4

        if "28.51 Растения как живые системы" in self.__final_scores:
            plants_sys_keywords = self.getKeySet(28.51)
            matches = len(self.__doc_words & plants_sys_keywords)
            if matches >= 5:
                self.__final_scores["28.51 Растения как живые системы"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.51 Растения как живые системы"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.51 Растения как живые системы"] *= 1.4

        if "28.52 Эволюционная биология растений" in self.__final_scores:
            plant_evol_keywords = self.getKeySet(28.52)
            matches = len(self.__doc_words & plant_evol_keywords)
            if matches >= 4:
                self.__final_scores["28.52 Эволюционная биология растений"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.52 Эволюционная биология растений"] *= 1.6

        if "28.54 Генетика растений" in self.__final_scores:
            plant_gen_keywords = self.getKeySet(28.54)
            matches = len(self.__doc_words & plant_gen_keywords)
            if matches >= 5:
                self.__final_scores["28.54 Генетика растений"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.54 Генетика растений"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.54 Генетика растений"] *= 1.4

        if "28.66 Морфология, анатомия и гистология животных" in self.__final_scores:
            animal_morph_keywords = self.getKeySet(28.66)
            matches = len(self.__doc_words & animal_morph_keywords)
            if matches >= 5:
                self.__final_scores["28.66 Морфология, анатомия и гистология животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.66 Морфология, анатомия и гистология животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.66 Морфология, анатомия и гистология животных"] *= 1.4

        if "28.69 Систематика животных" in self.__final_scores:
            animal_syst_keywords = self.getKeySet(28.69)
            matches = len(self.__doc_words & animal_syst_keywords)
            if matches >= 4:
                self.__final_scores["28.69 Систематика животных"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.69 Систематика животных"] *= 1.6

        if "28.68 Экология и география животных" in self.__final_scores:
            animal_eco_keywords = self.getKeySet(28.68)
            matches = len(self.__doc_words & animal_eco_keywords)
            if matches >= 5:
                self.__final_scores["28.68 Экология и география животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.68 Экология и география животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.68 Экология и география животных"] *= 1.4

        if "28.59 Систематика растений" in self.__final_scores:
            plant_syst_keywords = self.getKeySet(28.59)
            matches = len(self.__doc_words & plant_syst_keywords)
            if matches >= 4:
                self.__final_scores["28.59 Систематика растений"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.59 Систематика растений"] *= 1.6

        if "28.64 Генетика животных" in self.__final_scores:
            animal_gen_keywords = self.getKeySet(28.64)
            matches = len(self.__doc_words & animal_gen_keywords)
            if matches >= 5:
                self.__final_scores["28.64 Генетика животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.64 Генетика животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.64 Генетика животных"] *= 1.4

        if "28.62 Эволюционная биология животных" in self.__final_scores:
            animal_evol_keywords = self.getKeySet(28.62)
            matches = len(self.__doc_words & animal_evol_keywords)
            if matches >= 5:
                self.__final_scores["28.62 Эволюционная биология животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.62 Эволюционная биология животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.62 Эволюционная биология животных"] *= 1.4

        if "28.63 Биология развития животных" in self.__final_scores:
            animal_dev_keywords = self.getKeySet(28.63)
            matches = len(self.__doc_words & animal_dev_keywords)
            if matches >= 5:
                self.__final_scores["28.63 Биология развития животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.63 Биология развития животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.63 Биология развития животных"] *= 1.4

        if "28.03 Биология развития" in self.__final_scores:
            dev_bio_keywords = self.getKeySet(28.03)
            matches = len(self.__doc_words & dev_bio_keywords)
            if matches >= 5:
                self.__final_scores["28.03 Биология развития"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.03 Биология развития"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.03 Биология развития"] *= 1.4

        if "28.65 Цитология животных" in self.__final_scores:
            animal_cyt_keywords = self.getKeySet(28.65)
            matches = len(self.__doc_words & animal_cyt_keywords)
            if matches >= 5:
                self.__final_scores["28.65 Цитология животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.65 Цитология животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.65 Цитология животных"] *= 1.4

        if "28.06 Общая морфология" in self.__final_scores:
            morph_keywords = self.getKeySet(28.06)
            matches = len(self.__doc_words & morph_keywords)
            if matches >= 5:
                self.__final_scores["28.06 Общая морфология"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.06 Общая морфология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.06 Общая морфология"] *= 1.4

        if "28.07 Физико-химическая биология" in self.__final_scores:
            physchem_keywords = self.getKeySet(28.07)
            matches = len(self.__doc_words & physchem_keywords)
            if matches >= 5:
                self.__final_scores["28.07 Физико-химическая биология"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.07 Физико-химическая биология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.07 Физико-химическая биология"] *= 1.4

        if "28.08 Биоэкология" in self.__final_scores:
            bioeco_keywords = self.getKeySet(28.08)
            matches = len(self.__doc_words & bioeco_keywords)
            if matches >= 5:
                self.__final_scores["28.08 Биоэкология"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.08 Биоэкология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.08 Биоэкология"] *= 1.4

        if "28.16 Палеозоология" in self.__final_scores:
            paleozoo_keywords = self.getKeySet(28.16)
            matches = len(self.__doc_words & paleozoo_keywords)
            if matches >= 4:
                self.__final_scores["28.16 Палеозоология"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.16 Палеозоология"] *= 1.6

        if "28.32 Эволюция вирусов" in self.__final_scores:
            virus_evol_keywords = self.getKeySet(28.32)
            matches = len(self.__doc_words & virus_evol_keywords)
            if matches >= 4:
                self.__final_scores["28.32 Эволюция вирусов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.32 Эволюция вирусов"] *= 1.6

        if "28.37 Физико-химическая биология вирусов" in self.__final_scores:
            virus_phys_keywords = self.getKeySet(28.37)
            matches = len(self.__doc_words & virus_phys_keywords)
            if matches >= 4:
                self.__final_scores["28.37 Физико-химическая биология вирусов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.37 Физико-химическая биология вирусов"] *= 1.6

        if "28.38 Экология вирусов" in self.__final_scores:
            virus_eco_keywords = self.getKeySet(28.38)
            matches = len(self.__doc_words & virus_eco_keywords)
            if matches >= 4:
                self.__final_scores["28.38 Экология вирусов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.38 Экология вирусов"] *= 1.6

        if "28.42 Эволюция микроорганизмов" in self.__final_scores:
            microb_evol_keywords = self.getKeySet(28.42)
            matches = len(self.__doc_words & microb_evol_keywords)
            if matches >= 4:
                self.__final_scores["28.42 Эволюция микроорганизмов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.42 Эволюция микроорганизмов"] *= 1.6

        if "28.45 Цитология микроорганизмов" in self.__final_scores:
            microb_cyt_keywords = self.getKeySet(28.45)
            matches = len(self.__doc_words & microb_cyt_keywords)
            if matches >= 5:
                self.__final_scores["28.45 Цитология микроорганизмов"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.45 Цитология микроорганизмов"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.45 Цитология микроорганизмов"] *= 1.4

        if "28.48 Экология микроорганизмов" in self.__final_scores:
            microb_eco_keywords = self.getKeySet(28.48)
            matches = len(self.__doc_words & microb_eco_keywords)
            if matches >= 5:
                self.__final_scores["28.48 Экология микроорганизмов"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.48 Экология микроорганизмов"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.48 Экология микроорганизмов"] *= 1.4

        if "28.49 Систематика микроорганизмов" in self.__final_scores:
            microb_syst_keywords = self.getKeySet(28.49)
            matches = len(self.__doc_words & microb_syst_keywords)
            if matches >= 4:
                self.__final_scores["28.49 Систематика микроорганизмов"] *= 1.9
            elif matches >= 2:
                self.__final_scores["28.49 Систематика микроорганизмов"] *= 1.6

        if "28.53 Биология развития растений" in self.__final_scores:
            plant_dev_keywords = self.getKeySet(28.53)
            matches = len(self.__doc_words & plant_dev_keywords)
            if matches >= 5:
                self.__final_scores["28.53 Биология развития растений"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.53 Биология развития растений"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.53 Биология развития растений"] *= 1.4

        if "28.58 Экология растений" in self.__final_scores:
            plant_eco_keywords = self.getKeySet(28.58)
            matches = len(self.__doc_words & plant_eco_keywords)
            if matches >= 5:
                self.__final_scores["28.58 Экология растений"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.58 Экология растений"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.58 Экология растений"] *= 1.4

        if "28.67 Физиология животных" in self.__final_scores:
            animal_phys_keywords = self.getKeySet(28.67)
            matches = len(self.__doc_words & animal_phys_keywords)
            if matches >= 5:
                self.__final_scores["28.67 Физиология животных"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.67 Физиология животных"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.67 Физиология животных"] *= 1.4

        if "28.70 Биология человека" in self.__final_scores:
            human_bio_keywords = self.getKeySet(28.70)
            matches = len(self.__doc_words & human_bio_keywords)
            if matches >= 5:
                self.__final_scores["28.70 Биология человека"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.70 Биология человека"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.70 Биология человека"] *= 1.4

        if "28.71 Антропология" in self.__final_scores:
            anthrop_keywords = self.getKeySet(28.71)
            matches = len(self.__doc_words & anthrop_keywords)
            if matches >= 5:
                self.__final_scores["28.71 Антропология"] *= 2.0
            elif matches >= 3:
                self.__final_scores["28.71 Антропология"] *= 1.7
            elif matches >= 1:
                self.__final_scores["28.71 Антропология"] *= 1.4

    def getfinal_scores(self):
        return self.__final_scores