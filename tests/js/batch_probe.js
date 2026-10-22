// Runs after the page's readable-names and batch sections have been defined in the same scope.
const head = "Compound,Tyre Age,Stint Number,Lap Number,Race Progress,Track Temperature (°C),Air Temperature,Humidity,Wind Speed";
const join = (rows) => rows.join("\n");
const attempt = (fn) => {
  try {
    return fn();
  } catch (e) {
    return "ERROR: " + e.message;
  }
};
const out = {};
out.quoted = parseCsv('a,"b,c","say ""hi"""\r\n1,2,3\r\n');
out.bom_semicolon = parseCsv("﻿a;b\n1;2");
out.blank_lines = parseCsv("a,b\n\n1,2\n   \n");
out.good = buildBatch(join([head, "MEDIUM,14,1,18,0.3,38,27,45,2.5"]).replace(/\n/g, "\r\n")).laps[0];
out.length_instead = buildBatch(
  "compound,tyre_age,stint_number,lap_number,race_length,track_temperature,air_temperature,humidity,wind_speed\nSOFT,3,1,20,50,30,20,40,1",
).laps[0].race_progress;
out.optional_kept = buildBatch(head + ",Previous Lap Delta,Rainfall\nSOFT,3,1,5,0.1,30,20,40,1,0.25,1").laps[0];
out.missing = attempt(() => buildBatch("Compound,Tyre Age\nSOFT,3"));
out.not_number = attempt(() => buildBatch(join([head, "SOFT,3,1,5,0.1,30,20,40,2", "SOFT,abc,1,5,0.1,30,20,40,2"])));
out.empty_required = attempt(() => buildBatch(join([head, "SOFT,,1,5,0.1,30,20,40,2"])));
out.hex_rejected = attempt(() => buildBatch(join([head, "SOFT,0x10,1,5,0.1,30,20,40,2"])));
out.no_rows = attempt(() => buildBatch(head));
out.unclosed = attempt(() => parseCsv('a,"b'));
out.csv = [csvCell("=1+1"), csvCell("-0.5"), csvCell("@x"), csvCell("a,b"), csvCell('q"q'), csvCell(0.25)];
console.log(JSON.stringify(out));
