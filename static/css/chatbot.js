$(document).ready(function() {
    const chatbotContainer = $('#chatbot-container');
    const chatLog = $('#chat-log');
    const userInput = $('#user-input');
    const sendButton = $('#send-button');
    const companyDescriptionInput = $('#company-description');
    const jobRoleInput = $('#job-role');

    let companyDescriptionFile;
    let jobRoleFile;
    let pdfsUploaded = false;

    companyDescriptionInput.on('change', function(e) {
        companyDescriptionFile = e.target.files[0];
    });

    jobRoleInput.on('change', function(e) {
        jobRoleFile = e.target.files[0];
    });

    $('#upload-pdfs').on('click', function() {
        if (!companyDescriptionFile || !jobRoleFile) {
            alert("Please select both PDF files before uploading.");
            return;
        }

        const formData = new FormData();
        formData.append('company_description', companyDescriptionFile);
        formData.append('job_role', jobRoleFile);

        $.ajax({
            url: '/upload_pdfs',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                alert(response.message);
                pdfsUploaded = true;
            },
            error: function(xhr, status, error) {
                alert("Error uploading PDFs: " + xhr.responseJSON.error);
            }
        });
    });

    sendButton.on('click', sendMessage);

    userInput.on('keypress', function(e) {
        if (e.which === 13) {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userInput.val().trim();
        if (userMessage) {
            if (!pdfsUploaded) {
                alert("Please upload PDF files before sending a message.");
                return;
            }

            appendMessage(userMessage, 'user-message');

            $.ajax({
                url: '/chatbot',  // Ensure this matches the Flask route
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ question: userMessage }), // Ensure the key matches the expected input in Flask
                success: function(response) {
                    appendMessage(response.answer, 'chatbot-message'); // Match the response format
                },
                error: function(xhr, status, error) {
                    appendMessage("Error: " + xhr.responseJSON.error, 'error-message');
                }
            });

            userInput.val('');
        }
    }
    function appendMessage(message, className) {
        const messageElement = $('<div>').text(message).addClass(className);
        chatLog.append(messageElement);
        chatLog.scrollTop(chatLog[0].scrollHeight);
    }

    // Add event listener to close chatbot container
    $('#close-chatbot').on('click', function() {
        chatbotContainer.hide();
    });

    // Show chatbot when chat icon is clicked
    $('#chat-icon').on('click', function() {
        chatbotContainer.show();
    });
});
