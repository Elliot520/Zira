#!/bin/bash
# Builds bin/ZiraCalendar.app (see ZiraCalendar.swift). Needs Xcode's command line tools (swiftc).
set -euo pipefail
cd "$(dirname "$0")/../.."
APP=bin/ZiraCalendar.app
rm -rf "$APP" && mkdir -p "$APP/Contents/MacOS"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>com.zira.calendar</string>
  <key>CFBundleName</key><string>Zira Calendar</string>
  <key>CFBundleDisplayName</key><string>Zira Calendar</string>
  <key>CFBundleExecutable</key><string>ZiraCalendar</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSCalendarsFullAccessUsageDescription</key><string>Zira reads your events to tell you about your day, and adds events when you ask it to.</string>
  <key>NSCalendarsUsageDescription</key><string>Zira reads your events to tell you about your day, and adds events when you ask it to.</string>
</dict></plist>
PLIST
swiftc -O -o "$APP/Contents/MacOS/ZiraCalendar" third_party/calendar/ZiraCalendar.swift
codesign --force --sign - "$APP"
echo "built $APP"
