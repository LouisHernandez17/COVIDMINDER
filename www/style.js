// ids key: value pairs represent a corresponding sidebar: mainpanel
// resize iterates through and 'snaps' mainpanel to sidepbars height. 
var ids = {
  "sidebar_us_mort": "#mainpanel_us_mort",
  "sidebar_us_test": "#mainpanel_us_test",
  "sidebar_us_hosp": "#mainpanel_us_hosp",
  "sidebar_us_db": "#mainpanel_us_db",
  "sidebar_us_cardio": "#mainpanel_us_cardio",
  "sidebar_ny_mort": "#mainpanel_ny_mort",
  "sidebar_ny_cases": "#mainpanel_ny_cases",
  "sidebar_ny_CoT": "#mainpanel_ny_CoT", 
  "sidebar_ny_det": "#mainpanel_ny_det", 
  "sidebar_ny_CoT_rates": "#mainpanel_ny_CoT_rates", 
  "sidebar_ny_race": "#mainpanel_ny_race", 
  "sidebar_us_mort_race": "#mainpanel_us_mort_race", 
  "sidebar_ct_race": "#mainpanel_ct_race"
};

function resize_plot() {
  for (var key in ids) {
  var sidebar1 = document.getElementById(key);
  if(sidebar1) {
    var positionInfo = sidebar1.getBoundingClientRect();
    var height = positionInfo.height;
    $(ids[key]).height(height);
  }
  }
}

//$("#mainPanel_us_mort").on("resize", resize_plot);
window.addEventListener("resize", resize_plot);
window.addEventListener("click", resize_plot);
resize_plot();
