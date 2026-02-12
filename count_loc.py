
import os

def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def main():
    extensions = {
        '.rs': 'Rust',
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.html': 'HTML',
        '.css': 'CSS',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C/C++ Header',
        '.json': 'JSON'
    }
    
    ignore_dirs = {
        'node_modules', 'target', '.git', '.agent', '__pycache__', 'venv', '.venv', '.idea', '.vscode', 'build', 'dist', '.cursor'
    }

    stats = {lang: {'files': 0, 'lines': 0} for lang in extensions.values()}
    total_files = 0
    total_lines = 0

    root_dir = '.'
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                lang = extensions[ext]
                filepath = os.path.join(root, file)
                lines = count_lines_in_file(filepath)
                
                stats[lang]['files'] += 1
                stats[lang]['lines'] += lines
                total_files += 1
                total_lines += 1

    print(f"{'Language':<20} {'Files':<10} {'Lines':<10}")
    print("-" * 40)
    
    # Sort by lines descending
    sorted_stats = sorted(stats.items(), key=lambda item: item[1]['lines'], reverse=True)
    
    for lang, data in sorted_stats:
        if data['lines'] > 0:
            print(f"{lang:<20} {data['files']:<10} {data['lines']:<10}")
            
    print("-" * 40)
    print(f"{'Total':<20} {total_files:<10} {total_lines:<10}")

if __name__ == "__main__":
    main()
