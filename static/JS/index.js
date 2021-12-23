var selection = "EtoU";
var urduTextArea = document.getElementById("urdu-input");
var engTextArea = document.getElementById("english-input");
var translationBtn = document.getElementById("translateBtn");
var source = engTextArea;
var target = urduTextArea;
var text;

function dropdownChange(){
    if(selection == "EtoU"){
        console.log("Change to urdu translation");
        selection = "UtoE";
        urduTextArea.disabled = false;
        engTextArea.disabled = true;
        urduTextArea.value = "";
        engTextArea.value = "";
        source = urduTextArea;
        target = engTextArea;
        // translationBtn.innerHTML = "Translate urdu to english";

    }else if(selection == 'UtoE'){
        console.log("Change to english translation");
        selection = "EtoU";
        urduTextArea.disabled = true;
        engTextArea.disabled = false;
        urduTextArea.value = "";
        engTextArea.value = "";
        source = engTextArea;
        target = urduTextArea;
        // translationBtn.innerHTML = "Translate english to urdu";
    }
}

function btnClick(){
    text = source.value;
    target.value = "Text will be displayed here";
    console.log(selection);
    if(selection === "EtoU"){
        console.log(selction);
    }else if(selection === "UtoE"){
        console.log(selection);
    }
    
}

