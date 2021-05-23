$(document).ready(function () {
    bsCustomFileInput.init();
})

async function uploadTrainFile() {
    let formData = new FormData();
    formData.append("train_file", inputGroupFile02.files[0]);
    await fetch('/train', {
        method: "POST",
        body: formData
    });
    alert('The file has been uploaded successfully.');
}

async function uploadEvaluateFile2() {
    let formData = new FormData();
    formData.append("evaluate_file", inputGroupFile01.files[0]);
    const response = await fetch('/evaluate', {
        method: "POST",
        body: formData
    });
    const messages = await response.json();
    return messages
    // alert('The file has been uploaded successfully.');
}

function uploadEvaluateFile() {
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
    });
}