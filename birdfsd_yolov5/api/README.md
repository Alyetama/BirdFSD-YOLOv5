# BirdFSD-YOLOv5 API

- Run locally

```sh
python api.py
```

## Example

```sh
curl "https://d.aibird.me/be6fcfd8.jpg" --output "demo.jpg"
curl -X POST "http://127.0.0.1:8000/predict" -F file="@demo.jpg"
```

```json
{
    "results": {
        "input_image": {
            "name": "demo.jpg",
            "hash": "63149b75381ce19b352357efb7698a66"
        },
        "labeled_image_url": "https://api-s3.aibird.me/api/32a9002b42c8.jpg",
        "predictions": {
            "0": {
                "confidence": 0.905,
                "name": "Brown-headed Cowbird (Molothrus ater)",
                "bbox": {
                    "xmin": 0.29556283354759216,
                    "ymin": 0.5663608312606812,
                    "xmax": 0.5031617879867554,
                    "ymax": 0.8657128214836121
                },
                "species_info": {
                    "gbif": {
                        "usageKey": 2484391,
                        "scientificName": "Molothrus ater (Boddaert, 1783)",
                        "canonicalName": "Molothrus ater",
                        "rank": "SPECIES",
                        "kingdom": "Animalia",
                        "phylum": "Chordata",
                        "order": "Passeriformes",
                        "family": "Icteridae",
                        "genus": "Molothrus",
                        "species": "Molothrus ater",
                        "synonym": false,
                        "class": "Aves"
                    }
                }
            }
        },
        "model": {
            "name": "BirdFSD-YOLOv5",
            "version": "1.0.0-alpha.5",
            "page": "https://github.com/bird-feeder/BirdFSD-YOLOv5/releases/tag/1.0.0-alpha.5"
        }
    }
}
```
