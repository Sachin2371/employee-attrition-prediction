// Password strength indicator
document.addEventListener('DOMContentLoaded', function() {
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm_password');
    const passwordStrength = document.getElementById('passwordStrength');
    const passwordMatch = document.getElementById('passwordMatch');
    const submitBtn = document.getElementById('submitBtn');

    if (passwordInput) {
        passwordInput.addEventListener('input', function() {
            const password = this.value;
            let strength = 0;

            // Length check
            if (password.length >= 8) strength += 25;

            // Lowercase check
            if (/[a-z]/.test(password)) strength += 25;

            // Uppercase check
            if (/[A-Z]/.test(password)) strength += 25;

            // Number/Special char check
            if (/[0-9!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) strength += 25;

            // Update progress bar
            passwordStrength.style.width = strength + '%';

            // Update color
            if (strength < 50) {
                passwordStrength.className = 'progress-bar bg-danger';
            } else if (strength < 75) {
                passwordStrength.className = 'progress-bar bg-warning';
            } else {
                passwordStrength.className = 'progress-bar bg-success';
            }
        });
    }

    // Password confirmation check
    if (confirmPasswordInput && passwordInput) {
        confirmPasswordInput.addEventListener('input', function() {
            const password = passwordInput.value;
            const confirmPassword = this.value;

            if (confirmPassword === '') {
                passwordMatch.textContent = '';
                passwordMatch.className = 'form-text';
            } else if (password === confirmPassword) {
                passwordMatch.textContent = 'Passwords match!';
                passwordMatch.className = 'form-text text-success';
            } else {
                passwordMatch.textContent = 'Passwords do not match!';
                passwordMatch.className = 'form-text text-danger';
            }
        });
    }

    // Form validation enhancement
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const inputs = this.querySelectorAll('input[required], select[required]');
            let valid = true;

            inputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    });

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.classList.add('fade-in');
        card.style.animationDelay = (index * 0.1) + 's';
    });
});