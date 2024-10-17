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

    // Handle PDF file selection
    companyDescriptionInput.on('change', function(e) {
        companyDescriptionFile = e.target.files[0];
    });

    jobRoleInput.on('change', function(e) {
        jobRoleFile = e.target.files[0];
    });

    // Handle PDF upload
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

    // Send message on button click or pressing Enter
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

            // Append user message to chat log
            appendMessage(userMessage, 'user-message');

            $.ajax({
                url: '/chatbot',  // Ensure this matches the Flask route
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ question: userMessage }), // Ensure the key matches expected input in Flask
                success: function(response) {
                    // Append chatbot response to chat log
                    appendMessage(response.answer, 'chatbot-message'); 
                },
                error: function(xhr, status, error) {
                    appendMessage("Error: " + xhr.responseJSON.error, 'error-message');
                }
            });

            // Clear user input
            userInput.val('');
        }
    }

    // Append message to chat log
    function appendMessage(message, className) {
        const messageElement = $('<div>').text(message).addClass(className);
        chatLog.append(messageElement);
        chatLog.scrollTop(chatLog[0].scrollHeight); // Scroll to bottom after adding message
    }

    // Close chatbot container
    $('#close-chatbot').on('click', function() {
        chatbotContainer.hide();
    });

    // Show chatbot when chat icon is clicked
    $('#chat-icon').on('click', function() {
        chatbotContainer.show();
    });
});
