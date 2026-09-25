# Discord Bot Manager für Home Assistant

Home-Assistant-App mit eigener deutscher Ingress-Oberfläche für den vorhandenen Discord-Bot. Unterstützt Home Assistant OS auf ARM64 (Raspberry Pi mit 64-Bit-OS) und AMD64.

## Installation aus der Ferne

1. Dieses Projekt in ein Git-Repository veröffentlichen. `env`, `.env`, `user_*.json`, `private/` und `.local/` sind absichtlich ausgeschlossen. Vor dem Veröffentlichen `git status` prüfen.
2. In Home Assistant: **Einstellungen → Apps → App-Store → ⋮ → Repositories**, deine Git-Repository-URL eintragen.
3. **Discord Bot Manager** installieren und starten. Der erste Build lädt die Python-Pakete; auf dem Pi kann das einige Minuten dauern.
4. **Weboberfläche öffnen**. Unter Einstellungen das Discord-Token eintragen. Der Einstiegspunkt ist `bot.py`.
5. Vorhandene Daten wie unten beschrieben importieren. Erst danach den Bot starten.

Das Projekt ist lokal vorbereitet; es wurde noch kein Remote-Repository angelegt oder veröffentlicht. Es werden keine Ports zum Host freigegeben. Zugriff erfolgt ausschließlich über Home-Assistant-Ingress; die Seitenleiste ist für Administratoren vorgesehen. Die tatsächlichen Ingress-Zugriffsrechte werden von Home Assistant verwaltet.

## Vorhandenen Bot und seine Daten übernehmen

Der vorhandene Python-Code liegt jetzt unter `discord_bot_manager/seed/`. Der Manager kopiert jede fehlende Datei nach `/share/discord-bot-manager/code/`. Bereits vorhandener Code wird bei Updates **nicht überschrieben**. Änderungen an der mitgelieferten Vorlage müssen bei bestehenden Installationen bewusst über den Editor oder Upload übernommen werden.

Die ursprünglichen JSON-Dateien bleiben unverändert im Projektordner. Das folgende Kommando erzeugt ein privates ZIP für den Umzug:

```sh
python scripts/export_existing_data.py
```

Bei gestopptem Bot in **Sicherungen → ZIP importieren** hochladen, Vorschau prüfen und Wiederherstellung bestätigen. Das ZIP enthält `storage/user_preferences.json` und die vorhandenen `storage/user_reports_*.json`, aber kein Token. Die lokale Datei `env` wird nicht automatisch übertragen. Trage ihr Discord-Token im geschützten Einstellungsfeld ein.

Im [Discord Developer Portal](https://discord.com/developers/applications) für den Bot **Message Content Intent** und **Server Members Intent** einschalten: Der vorhandene Bot aktiviert beide. Bei der Einladung die Scopes `bot` und `applications.commands` sowie die benötigten Kanalrechte (Nachrichten lesen/senden, Dateien anhängen, Nachrichtenverlauf lesen) wählen. Keine Administratorrechte erforderlich. Die Oberfläche zeigt Prozess- und Discord-Verbindungsstatus getrennt; eigene Bots ohne Statusmeldung zeigen „Unbekannt“.

## Funktionen

- Start/Stopp/Neustart, optionaler Autostart, maximal drei Wiederanläufe nach einem Absturz mit 5/10/15 Sekunden Wartezeit. Manuelles Stoppen verhindert weitere Wiederanläufe.
- Dateien und Ordner erstellen, Dateien hoch-/herunterladen, umbenennen und bearbeiten. Der Texteditor unterstützt UTF-8 bis 2 MiB. Änderungen sind nur bei gestopptem Bot möglich.
- Papierkorb mit Wiederherstellung und ausdrücklich bestätigtem endgültigem Löschen einzelner Dateien bzw. leerer Ordner.
- Live-Logs mit Suche und dauerhaft gespeicherten Logdateien. Bekannte Token und konfigurierte Umgebungswerte werden aus Manager-Logs maskiert.
- Paketinstallation über `code/requirements.txt` in einer eigenen Bot-Umgebung; Fortschritt und Fehler stehen in den Logs.
- ZIP-Sicherungen von Code, Daten und Einstellungen. Wiederherstellung mit Vorschau, Prüfsumme und zusätzlicher Sicherung vor dem Überschreiben. Zusätzliche Dateien bleiben erhalten.

## Speicher und Sicherheit

| Pfad | Inhalt |
| --- | --- |
| `/share/discord-bot-manager/code/` | Python-Code und requirements.txt |
| `/share/discord-bot-manager/storage/` | Bot-Daten; Arbeitsverzeichnis und `BOT_STORAGE_DIR` |
| `/share/discord-bot-manager/logs/` | Dauerhafte Manager-/Bot-/Installationslogs |
| `/share/discord-bot-manager/backups/` | Manuelle Sicherungen und importierte ZIP-Dateien |
| `/share/discord-bot-manager/trash/` | Über die Oberfläche entfernte Dateien |
| `/data/settings.json` | Token und Einstellungen, Dateimodus 0600 |
| `/data/python-…/` | Technische Bot-Python-Umgebung |

Der Manager bereinigt diese Daten nicht automatisch. Nur unvollständige interne Schreibdateien werden nach Fehlern entfernt. Bei weniger als 1 GiB oder 5 % freiem Speicher erscheint eine Warnung. Schreibfehler werden angezeigt; Dateien werden per temporärer Datei und atomarem Ersetzen gespeichert. Ein Restore ist pro Datei atomar, aber keine Transaktion über das gesamte Archiv: Bei einem Schreibfehler bleibt die vorherige Sicherung zur erneuten Wiederherstellung erhalten.

Der bestehende Bot entfernt selbst temporär erzeugte Diagramme nach dem Versand und kann per Discord-Befehl Berichte ändern/löschen. Dieses bestehende Verhalten bleibt erhalten. Die Aufbewahrungsgarantie des Managers ist kein Schutz vor eigenem Python-Code, Paketinstallationsskripten oder Datenträgerausfällen.

Bot und Manager haben getrennte Python-Umgebungen, aber sind **keine Sicherheits-Sandbox gegeneinander**. Nur vertrauenswürdigen Code/Pakete ausführen. Weder Supervisor-API noch Docker-Socket oder privilegierter Hostzugriff sind aktiviert; Supervisor-Umgebungsvariablen werden nicht an den Bot weitergegeben. Der Dateimanager blockiert absolute Pfade, Traversal und Symlinks. Gegen absichtlich gleichzeitig manipulierte Dateisysteme durch fremde Prozesse bietet er keine vollständige Sandbox.

Backups enthalten das Token im Klartext und sind vertraulich. Tokenmaskierung betrifft die vom Manager erfassten Ausgaben; vom eigenen Bot direkt geschriebene Dateien können Geheimnisse enthalten. Das Token-Feld wird nie mit dem gespeicherten Wert an den Browser zurückgegeben. Der Ingress-Endpunkt lässt nur die Supervisor-Adresse `172.30.32.2` zu; schreibende API-Anfragen benötigen zusätzlich einen CSRF-Token.

Home-Assistant-Sicherungen müssen die **App und den gemeinsam verwendeten share-Ordner** einschließen. Die technische Laufzeit darf bei Updates in einem neuen Verzeichnis erstellt werden; die Oberfläche bleibt erreichbar. Zusätzliche Pakete dann über die Schaltfläche erneut aus der gespeicherten requirements.txt installieren. Alte Laufzeitverzeichnisse werden nicht automatisch entfernt.

## Lokal entwickeln und prüfen

```sh
python -m venv .venv
.venv/bin/pip install -r discord_bot_manager/manager/requirements.txt pytest
.venv/bin/pytest -q
MANAGER_LOCAL=1 BOT_ROOT="$PWD/.local/share" MANAGER_DATA="$PWD/.local/data" BOT_SEED="$PWD/discord_bot_manager/seed" .venv/bin/python discord_bot_manager/manager/app.py
```

Lokal: `http://127.0.0.1:8099`. Der Entwicklungsmodus bindet ausschließlich an Loopback. Für den Bot zuerst über die Oberfläche die Abhängigkeiten installieren. Die Oberfläche verwendet Polling im Abstand von 2,5 Sekunden. Upload-Limit: 256 MiB; Wiederherstellungsarchive maximal 1 GiB entpackt und 10.000 Einträge. Es gibt noch kein interaktives Terminal und keine Mehrbot-Unterstützung.

## Abnahme auf Home Assistant OS

1. Daten importieren, Bot starten, gespeicherte JSON-Dateien/Prüfsummen prüfen.
2. Bot, App und anschließend Home Assistant OS neu starten; Prüfsummen vergleichen (Bot dabei keine Daten ändern lassen).
3. App-Version erhöhen und ein Update installieren; Code und Daten müssen erhalten bleiben.
4. Bot gezielt mit Syntaxfehler starten: Oberfläche bleibt erreichbar, Log zeigt Ursache, Neustartversuche sind begrenzt.
5. Mit gestopptem Bot Backup und Wiederherstellung prüfen; Discord-Anmeldung, Intents und Befehle prüfen.

Container-Build, ARM64-Laufzeit, Ingress und Geräte-/App-Updates müssen auf dem Zielsystem geprüft werden. Lokale Tests ersetzen diese Abnahme nicht.

Grundlagen: [App-Konfiguration](https://developers.home-assistant.io/docs/apps/configuration/), [Ingress](https://developers.home-assistant.io/docs/apps/presentation/#ingress), [Repository](https://developers.home-assistant.io/docs/apps/repository/).
