document.addEventListener('DOMContentLoaded', function() {
    const getTestBtn = document.getElementById('getTestBtn');
    const postTestBtn = document.getElementById('postTestBtn');
    const postInput = document.getElementById('postInput');
    const getResult = document.getElementById('getResult');
    const postResult = document.getElementById('postResult');
    
    // GET请求测试
    getTestBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/api/hello');
            const data = await response.json();
            getResult.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
            getResult.textContent = 'Error: ' + error.message;
        }
    });
    
    // POST请求测试
    postTestBtn.addEventListener('click', async function() {
        const inputText = postInput.value.trim();
        if (!inputText) {
            postResult.textContent = 'Please enter some text first';
            return;
        }
        
        try {
            const response = await fetch('/api/echo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: inputText,
                    timestamp: new Date().toISOString()
                })
            });
            const data = await response.json();
            postResult.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
            postResult.textContent = 'Error: ' + error.message;
        }
    });
});