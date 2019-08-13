import hawks

gen = hawks.create_generator("validation.json")

gen.run()

gen.save_stats_csv("validation.csv")