# README – Real-ESRGAN (Super-Resolution)

## Projektbeschreibung

Dieses Repository beschreibt den Einsatz von **Real-ESRGAN** zur Bildhochskalierung im Rahmen der Bachelorarbeit im Studiengang Wirtschaftsinformatik an der Hochschule für Technik und Wirtschaft Berlin (HTW Berlin).

Ziel des Projekts ist die Erstellung einer **Nutzungs- und Leistungsdokumentation für GPU-gestützte Leihlaptops**. Real-ESRGAN dient dabei als praxisnahes Einstiegsszenario für Studierende sowie als Werkzeug zur **Leistungsbewertung der GPU-Hardware** unter realistischen Bedingungen.

---

## Art der Daten

1. **Eingabebilder (Referenzdaten)**

   * Niedrig aufgelöste oder unscharfe Bilder
   * Unterschiedliche Szenarien (Natur, Stadt, Figuren, Tiere)

2. **Ergebnisbilder (Upscaling-Ergebnisse)**

   * Mit Real-ESRGAN hochskalierte Bilder
   * Vergrößerte Auflösung und verbesserte Detaildarstellung

3. **Logdaten**

   * CSV-Dateien mit Laufzeit- und Hardwarekennzahlen

**Sprache:** Deutsch

**Methode:**
Vorher–Nachher-Vergleich zur Bewertung von Bildqualität und Systemleistung.

---

## Datenursprung

* **Autor:** Studierender der HTW Berlin, Wirtschaftsinformatik

* **Eingabebilder (Testdaten):**
  Die verwendeten Eingabebilder wurden aus öffentlich zugänglichen Internetquellen bezogen und dienen ausschließlich als **technische Test- und Referenzdaten** für die Leistungsbewertung der GPU-gestützten Leihlaptops. Eine inhaltliche Analyse der Bilder erfolgte nicht.

* **Ergebnisbilder (Bounding Boxes):**
  Die Ergebnisbilder wurden durch die Anwendung des Real-ESRGAN-Modells vom Autor selbst erzeugt und stellen **abgeleitete Daten** dar.

* **Log- und Messdaten:**
  Die Performance-Logs wurden vollständig selbst erhoben.

* **Lizenz:**

  * **Code & Logs:** BSD 3-Clause License
  * **Bilder:** Verwendung als Test- und Analyseartefakte

---

## Datenformate und -umfang

* Bilder: *.webp*, *.jfif* (mehrere KB pro Datei)
* Logs: *.csv* (1-2 KB)

---

## Qualitätssicherung

* Vergleich identischer Bilder mit und ohne Upscaling
* Sichtprüfung auf Artefakte
* Reproduzierbare Parametereinstellungen

---

## Ordnerstruktur

```
/
├─ images_input/
├─ images_output/
├─ logs/
└─ README.md
```

---

## Hinweise zur Nachnutzung

Die Nutzung und Weiterverbreitung ist unter den Bedingungen der BSD 3-Clause License zulässig.
