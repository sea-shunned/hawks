from pathlib import Path
import hawks

gen = hawks.create_generator("validation.json")

gen.run()

hawks.utils.df_to_csv(
    gen.stats,
    Path.cwd(),
    "validation"
)