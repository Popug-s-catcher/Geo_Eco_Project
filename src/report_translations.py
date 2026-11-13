import json
from pathlib import Path

TRANSLATIONS_DIR = Path(__file__).parent / 'translations'


def load_translations(translations_dir=TRANSLATIONS_DIR):
    translations = {}
    value_translations = {}
    date_formats = {}

    for path in sorted(Path(translations_dir).glob('*.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        translations[path.stem] = data['phrases']
        value_translations[path.stem] = data.get('values', {})
        date_formats[path.stem] = data.get(
            'dates',
            {'format': '{day:02d}.{month_num:02d}.{year}'}
        )

    return translations, value_translations, date_formats


TRANSLATIONS, VALUE_TRANSLATIONS, DATE_FORMATS = load_translations()