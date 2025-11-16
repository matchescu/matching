from matchescu.extraction import Traits


MAGELLAN_CONFIG = {
    "abt-buy": {
        "traits": Traits().string(["name", "description"]).currency(["price"]),
        "batch_size": 64,
        "epochs": 15,
    },
    "amazon-google": {
        "traits": Traits().string(["title", "manufacturer"]).currency(["price"]),
        "batch_size": 64,
        "epochs": 15,
    },
    "beer": {
        "traits": Traits().string(["Beer_Name", "Brew_Factory_Name", "Style"]),
        "batch_size": 32,
        "epochs": 40,
    },
    "dblp-acm": {
        "traits": Traits().string(["title", "authors", "venue"]).int(["year"]),
        "batch_size": 64,
        "epochs": 15,
    },
    "dblp-scholar": {
        "traits": Traits().string(["title", "authors", "venue", "year"]),
        "batch_size": 64,
        "epochs": 15,
    },
    "fodors-zagat": {
        "traits": Traits()
        .string(["name", "addr", "city", "phone", "type"])
        .int(["class"]),
        "batch_size": 32,
        "epochs": 40,
    },
    "itunes-amazon": {
        "traits": Traits().string(
            [
                "Song_Name",
                "Artist_Name",
                "Album_Name",
                "Genre",
                "Price",
                "CopyRight",
                "Time",
                "Released",
            ]
        ),
        "batch_size": 32,
        "epochs": 40,
        "learning_rate": 1e-5,
    },
    "walmart-amazon": {
        "traits": Traits()
        .string(["title", "category", "brand", "modelno"])
        .currency(["price"]),
        "batch_size": 64,
        "epochs": 15,
    },
}
