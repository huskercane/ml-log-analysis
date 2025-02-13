$(document).ready(function () {
    bsCustomFileInput.init();
})

async function uploadTrainFile() {
    $('#progressModal').modal('show');
    let formData = new FormData();
    formData.append("train_file", inputGroupFile02.files[0]);
    await fetch('/train', {
        method: "POST",
        body: formData
    });
    $('#progressModal').modal('toggle');
    alert('The file has been uploaded successfully.');
}

async function uploadEvaluateFile2() {
    let formData = new FormData();
    formData.append("evaluate_file", inputGroupFile01.files[0]);
    const response = await fetch('/evaluate', {
        method: "POST",
        body: formData
    });
    return await response.json()
    // alert('The file has been uploaded successfully.');
}

function uploadEvaluateFile() {
    $('#progressModal').modal('show');
    clearTable();
    uploadEvaluateFile2().then(messages => {
        let
            tableData = document.getElementById("results");

//set header of table
        let table = `
<table class="table table-striped" id = "myTable">
  <thead>
    <tr>
      <th scope="col">#</th>
      <th scope="col">Message</th>
    </tr>
  </thead>
  <tbody>
  `;
        //create//append rows
        for (i = 0; i < messages.length; i++) {
            table = table +
                `<tr>
      <th scope="row">${i}</th>
      <td>${messages[i]}</td>
    </tr>`
        }
//close off table
        table = table +
            `</tbody>
  </table>`
        ;

        tableData.innerHTML = table;
        $('#progressModal').modal('toggle');
    });
}

function clearTable() {
    let myNode = document.getElementById("results");
    while (myNode.firstChild) {
        myNode.removeChild(myNode.firstChild);
    }
}